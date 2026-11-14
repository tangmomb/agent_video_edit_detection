"""Entraînement de train_model sur un GPU Modal (TRAIN_GPU=modal dans .env), côté VPS.

Le VPS ne construit pas la matrice d'embeddings : il envoie les images (vidéo, numéro d'image), les
étiquettes, les plis déjà rééchantillonnés et pondérés (Lab._balance, Lab._weights) et le modèle ;
le GPU lit les embeddings dans le volume Modal (feature_volume.py), entraîne avec train_torch.py et
renvoie les probabilités hors pli et les paramètres de chaque modèle de pli. Ici, ces paramètres
redeviennent des objets scikit-learn ordinaires (rebuild) : model.pkl, la production, les pages et
final_test prédisent sur CPU comme avant.
"""

import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from video_source import retry

from .encoders import DERIVED, VIEWS

APP = "agent-training"


def enabled() -> bool:
    return os.environ.get("TRAIN_GPU") == "modal"


def cache_spec(spec: str) -> tuple[str, int, int]:
    """Embedding "extracteur:vue" -> (dossier du cache, parts par image, parts à sauter) : une vue
    dérivée (quads, tb) est la fin de la vue en cache (grid2, bands), comme dans get_features."""
    ext, _, view = spec.partition(":")
    view = view or "full"
    if view in DERIVED:
        parent = DERIVED[view]
        return f"{ext}__{parent}", VIEWS[parent], 1
    return f"{ext}__{view}", 1, 0


def incomplete(feat_dir: Path, specs: list[tuple[str, int, int]], sids: list[str],
               frames: list[int]) -> dict[str, set[str]]:
    """Dossier de cache -> vidéos dont une image demandée n'est pas encore calculée (fichier absent
    ou numéro manquant). Ne lit que les numéros d'images de chaque fichier, pas les embeddings."""
    wanted: dict[str, set[int]] = {}
    for s, f in zip(sids, frames):
        wanted.setdefault(s, set()).add(int(f))
    out: dict[str, set[str]] = {}
    for dirname in dict.fromkeys(d for d, _, _ in specs):
        for s, fs in wanted.items():
            path = Path(feat_dir) / dirname / f"{s}.npz"
            if path.exists():
                with np.load(path) as z:
                    if fs <= set(z["ids"].tolist()):
                        continue
            out.setdefault(dirname, set()).add(s)
    return out


def rebuild(model: str, params: dict, fold: dict, mlp_alpha: float):
    """Pipeline scikit-learn (StandardScaler [+ PCA] + classifieur) d'un modèle de pli entraîné par
    train_torch, identique à celui que Lab._make aurait ajusté."""
    sc_p = fold["prep"]["scaler"]
    d = len(sc_p["mean"])
    sc = StandardScaler()
    sc.mean_, sc.var_, sc.scale_ = (np.asarray(sc_p[k], dtype=np.float64) for k in ("mean", "var", "scale"))
    sc.n_features_in_, sc.n_samples_seen_ = d, np.int64(sc_p["n"])
    steps, width = [sc], d
    pca_p = fold["prep"].get("pca")
    if pca_p:
        pca = PCA(n_components=int(params["pca"]), random_state=0)
        pca.mean_ = np.asarray(pca_p["mean"], dtype=np.float64)
        pca.components_ = np.asarray(pca_p["components"], dtype=np.float64)
        pca.explained_variance_ = np.asarray(pca_p["explained_variance"], dtype=np.float64)
        pca.explained_variance_ratio_ = np.asarray(pca_p["explained_variance_ratio"], dtype=np.float64)
        pca.singular_values_ = np.asarray(pca_p["singular_values"], dtype=np.float64)
        pca.noise_variance_ = float(pca_p["noise_variance"])
        pca.n_components_, pca.n_samples_, pca.n_features_in_ = int(pca_p["k"]), int(pca_p["n"]), d
        steps.append(pca)
        width = int(pca_p["k"])
    if model == "logreg":
        lr_p = fold["logreg"]
        clf = LogisticRegression(C=float(params.get("C", 0.1)), max_iter=3000, class_weight="balanced")
        clf.coef_ = np.asarray(lr_p["coef"], dtype=np.float64).reshape(1, -1)
        clf.intercept_ = np.asarray([float(lr_p["intercept"])])
        clf.n_iter_ = np.asarray([int(lr_p["n_iter"])], dtype=np.int32)
    else:
        m_p = fold["mlp"]
        clf = MLPClassifier(hidden_layer_sizes=tuple(params.get("hidden", [256])),
                            alpha=float(params.get("alpha", mlp_alpha)), random_state=0)
        clf.coefs_ = [np.asarray(c, dtype=np.float32) for c in m_p["coefs"]]
        clf.intercepts_ = [np.asarray(b, dtype=np.float32) for b in m_p["intercepts"]]
        clf.n_layers_ = len(clf.coefs_) + 1
        clf.n_outputs_ = 1
        clf.out_activation_ = "logistic"
        clf._label_binarizer = LabelBinarizer().fit([0, 1])
        clf.n_iter_, clf.t_ = int(m_p["n_iter"]), int(m_p["t"])
    clf.classes_ = np.array([0, 1])
    clf.n_features_in_ = width
    return make_pipeline(*steps, clf)


def pack_key(payload: dict) -> str:
    """Clé de la matrice assemblée : mêmes images et mêmes embeddings, même matrice (l'embedding
    d'une image ne change pas), réutilisée par les train_model suivants."""
    import hashlib
    import json

    h = hashlib.sha1(json.dumps([payload["namespace"], payload["specs"], payload["sids"],
                                 [int(f) for f in payload["frames"]]]).encode())
    return h.hexdigest()[:20]


def train(payload: dict) -> dict:
    """Assemblage de la matrice sur un conteneur CPU Modal (pack), puis entraînement sur GPU
    (train) ; une panne ou un manque de place met l'entraînement en pause (un run peut attendre),
    comme les embeddings. Renvoie {"missing": [...]} si le volume n'a pas toutes les images."""
    import modal

    data = {k: payload[k] for k in ("namespace", "specs", "sids", "frames")}
    data["key"] = pack_key(payload)
    packed = retry.forever(lambda: modal.Function.from_name(APP, "pack").remote(data), "assemblage CPU")
    if packed.get("missing"):
        return packed
    job = {"pack": packed["pack"], "y": payload["y"], "plan": payload["plan"]}
    out = retry.forever(lambda: modal.Function.from_name(APP, "train").remote(job), "entraînement GPU")
    out["timing"] = {**{k: v for k, v in packed.items() if k not in ("pack",)}, **(out.get("timing") or {})}
    return out
