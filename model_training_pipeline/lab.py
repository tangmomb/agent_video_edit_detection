"""Laboratoire du classifieur d'images « la frame contient un ajout de montage » : toutes les
opérations sur les données, les modèles et les évaluations, sans aucune dépendance aux agents.
Les agents (package agents/) n'exposent qu'un sous-ensemble de ces méthodes comme outils.

Règles :
- données : vraies vidéos YouTube (video_source.youtube), une image par seconde ; cible binaire
  label_any (au moins un ajout de montage sur l'image), donnée par l'accord unanime des trois LLM
  d'annotators.ANNOTATORS. Une image sans accord n'est ni entraînée ni mesurée ;
- métrique principale : rappel à précision fixée (90 % par défaut, 95 % avant le 03/11/2026), seuil choisi sur val ;
- la collecte (generate_data) ignore les splits ; un split est un découpage dev/test nommé et
  figé, fait par chaîne YouTube (make_split, stockés dans data/splits/) : une chaîne garde le
  même habillage d'une vidéo à l'autre, elle ne doit pas être des deux côtés. Les vidéos collectées
  après sa création n'y entrent pas. Chaque expérience retient son split : on ne compare que des
  expériences faites sur le même split. L'ancien découpage « hash » n'est plus utilisable : il
  grossissait à chaque collecte, donc son val changeait sans changer de nom ;
- *val* est une validation croisée par chaîne sur dev (FOLDS plis) : chaque image de dev est
  prédite par un modèle qui n'a vu aucune vidéo de sa chaîne. Un val fixe ne comptait que 5 à 13
  chaînes (banc b1) : trop peu pour comparer des modèles, et son seuil ne tenait pas la précision
  sur d'autres chaînes. Le seuil est le seuil prudent de metrics (bootstrap par chaîne) ; le
  modèle enregistré est la moyenne des modèles de plis (FoldEnsemble). Les anciens splits train/val/test se lisent
  comme dev (train + val) / test ;
- les décisions se prennent sur *val* ; *test* n'est accessible que par final_test, en
  nombre limité par session ;
- chaque entraînement est enregistré dans model_results/<exp_id>/ (à la racine du projet).
"""

import base64
import io
import json
import sys
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from video_source import annotators, youtube
from video_source.atomic import sync_dir, write_atomic
from video_source.config import DATA_DIR, LAB_DIR, SAMPLES_DIR

from . import batch, describe, metrics
from .features import extraction_cost, get_features
from .frames import FPS, FRAMES_DIR, SKIPPED_SHARE, annotate, build, load_index, source_key, to_annotate

EXP_DIR = LAB_DIR / "model_results"
SPLITS_DIR = DATA_DIR / "splits"
# Clips retirés des splits créés ensuite (vidéo défectueuse, doublon) : exclude_clips, à la main (CLI exclude).
EXCLUDED = DATA_DIR / "excluded.json"
# Découpage d'avant make_split, non figé : refusé pour entraîner et tester.
LEGACY_SPLIT = "hash"
SPLITS = ("dev", "test")
FOLDS = 5  # plis de la validation croisée sur dev
# Listes par chaîne des outils (dataset_info) : au plus TOP_GROUPS
# entrées, le reste en un total, pour que les réponses ne grandissent pas avec le dataset.
TOP_GROUPS = 15
# Au-delà, le test ne gagne guère en précision : le reste va en dev.
TEST_MAX_CHAINS = 150
HISTORY_QUERIES = 30  # data_requested_history : requêtes listées, les plus productives
# list_experiments : une ligne par expérience qui compte sur le split récent, les autres en un
# total, pour que la réponse ne grandisse pas avec leur nombre
LISTED_BEST = 5  # les meilleures en AP par chaîne
LISTED_RECENT = 3  # les dernières, sur lesquelles le Model Researcher enchaîne
BRIEF_BEST = 3  # version de l'orchestrateur : les meilleures seulement
# Choix du modèle de production : deux AP par chaîne à moins de AP_TIE l'une de l'autre comptent
# comme égales, et le meilleur rappel val à la précision visée l'emporte. Sur split_003, exp_072
# (AP 0.9358, rappel 0.850) passait devant exp_079 (0.9351, 0.863) pour un écart d'AP sans portée.
# Fixé à l'œil (01/11/2026), pas mesuré sur le banc.
AP_TIE = 0.005
# Chaînes de test communes en dessous desquelles on ne compare pas un candidat à la production.
MIN_CHAINS_VS_PROD = 20
# Caractéristiques des vidéos sur lesquelles make_split peut équilibrer les splits.
STRATIFY_KEYS = ("orientation", "clip_vide")
# hgb (HistGradientBoosting) retiré le 26/10/2026 : sur les 20 essais du banc b2, jamais le meilleur
# ni sur val ni sur les chaînes cachées (rappel idéal caché 0.14 en moyenne contre 0.51 en logreg),
# pour 270 fois le temps d'entraînement ; des arbres conviennent mal à des embeddings denses.
MODELS = ("logreg", "mlp")
# mlp (03/11/2026) : arrêté à la meilleure AP par chaîne sur 15 % des chaînes de train mises de
# côté (avant : 10 % d'images au hasard, de chaînes vues), au plus 200 époques, patience 10 ;
# classes équilibrées ; alpha par défaut 1 (avant 0.001). Rejoué sur exp_094/131/149 : AP par
# chaîne +1 à +4 points sur split_003, mais toujours 4 à 5 points sous la logreg équivalente.
MLP_HOLDOUT, MLP_MAX_EPOCHS, MLP_PATIENCE, MLP_ALPHA = 0.15, 200, 10, 1.0
# Groupes d'images de train qu'on peut rééchantillonner (paramètre balance) ou pondérer (weights) :
# positif / negatif = images avec / sans ajout ; clip_vide = images des vidéos sans aucune image
# avec ajout ; orientation.
BALANCE_GROUPS = ("positif", "negatif", "clip_vide", "portrait", "paysage")
MAX_FACTOR = 10  # facteur maximal de balance et de weights


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _round_robin(items: list, key) -> list:
    """Items dans leur ordre, mais un par groupe (key) à tour de rôle."""
    groups: dict = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    out, queues = [], list(groups.values())
    while queues:
        out += [q.pop(0) for q in queues]
        queues = [q for q in queues if q]
    return out


def _spread(items: list, n: int) -> list:
    """n items régulièrement espacés dans la liste, sans les extrémités (souvent les images noires
    de début et de fin de vidéo) ; tous s'il y en a moins."""
    if len(items) <= n:
        return list(items)
    return [items[int((k + 0.5) * len(items) / n)] for k in range(n)]


def _image_blocks(path: Path, caption: str) -> list[dict]:
    """Légende + image JPEG en base64, au format des blocs renvoyés aux agents."""
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=80)
    return [{"type": "text", "text": caption},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(buf.getvalue()).decode()}}]


def _compact_val(val: dict) -> dict:
    """Val pour les agents : AP par chaîne (choix du modèle de production ; l'AP globale à défaut),
    rappel et erreurs au point de précision par défaut (90 %). Ni la précision (au moins 90 % par
    construction), ni l'AUC et le seuil, qui ne décident de rien ; les volumes du val sont ceux du
    split."""
    out = ({"ap_par_chaine": val["ap_par_chaine"]} if val.get("ap_par_chaine") is not None
           else {"ap": val["ap"]})
    op = val[f"rappel_a_precision_{metrics.DEFAULT_PRECISION}"]
    if not isinstance(op, dict):
        return {**out, "p_visee": op}  # "précision non atteinte"
    return {**out, "recall": op["recall"], "fp": op["fp"], "fn": op["fn"]}


def _config(e: dict) -> dict:
    """Ce qu'il faut pour refaire l'expérience : embeddings, modèle, hyperparamètres,
    rééchantillonnage et pondération (ces deux derniers seulement s'ils sont utilisés)."""
    out = {"embeddings": e["features"], **{k: e[k] for k in ("model", "params") if k in e}}
    out.update({k: e[k] for k in ("balance", "weights") if e.get(k)})
    return out


def _score(e: dict) -> float:
    """AP par chaîne en val (le critère du modèle de production), à défaut l'AP."""
    v = e["val"]
    return v.get("ap_par_chaine") if v.get("ap_par_chaine") is not None else v["ap"]


def _one_line(e: dict, test: str | None = None) -> str:
    """Expérience en une ligne, sans hypothèse : sa config (de quoi la refaire ou ne pas la
    refaire), son résultat val à la précision par défaut et son test final (test : statut, pour une
    expérience d'un ancien split dont seul le statut compte)."""
    c = _config(e)
    parts = ["+".join(c["embeddings"]), " ".join([c["model"], *(f"{k}={v}" for k, v in c.get("params", {}).items())])]
    parts += [f"{k} " + " ".join(f"{g}×{x}" for g, x in c[k].items()) for k in ("balance", "weights") if k in c]
    v = _compact_val(e["val"])
    res = f"AP/chaîne {_score(e)}, " + (f"rappel {v['recall']} (FP {v['fp']}, FN {v['fn']})"
                                         if "recall" in v else f"{metrics.DEFAULT_PRECISION * 100:g} % : {v['p_visee']}")
    if test:
        res += f" ; test {test}"
    elif "final_test" in e:
        t = e["final_test"]["test"]
        res += f" ; test : rappel {round(t['recall'], 4)}, précision {round(t['precision'], 4)}"
    return ", ".join(parts) + " → " + res


def _split_line(split_id: str, exps: list[dict], test_status) -> dict:
    """Ancien split : nombre d'expériences et la meilleure (AP par chaîne en val, le critère du modèle
    de production), en une ligne avec sa config pour être refaite sur le split récent et le statut
    de son test final (test_status : id -> réussi / échoué / non testé)."""
    best = max(exps, key=_score)
    return {"split_id": split_id, "experiences": len(exps),
            "meilleure": {best["id"]: _one_line(best, test_status(best["id"]))}}


def _best_by_rule(rows: list[dict]) -> dict | None:
    """Règle de choix du meilleur modèle (lignes de Lab._ranked) : parmi ceux dont l'AP par chaîne
    est à moins de AP_TIE de la meilleure, celui de meilleur rappel val ; None sans ligne."""
    if not rows:
        return None
    top = max(r["ap_par_chaine"] or 0.0 for r in rows)
    return max((r for r in rows if (r["ap_par_chaine"] or 0.0) >= top - AP_TIE),
               key=lambda r: (r["val_recall"], r["ap_par_chaine"] or 0.0, r["ap"], r["auc"], -r["ms_par_frame"]))


class LabeledRows:
    """Matrice des seules lignes étiquetées, lue comme si elle couvrait tout l'index : X[idx] avec
    des numéros de ligne de Lab.rows. Lire une ligne non étiquetée est une erreur."""

    def __init__(self, data: np.ndarray, rows: np.ndarray, n_rows: int):
        self.data = data
        self._pos = np.full(n_rows, -1, dtype=np.int64)
        self._pos[rows] = np.arange(len(rows))
        self.shape = (n_rows, data.shape[1])

    def __getitem__(self, idx) -> np.ndarray:
        pos = self._pos[np.asarray(idx, dtype=int)]
        if (pos < 0).any():
            raise IndexError("ligne sans étiquette : pas d'embeddings chargés")
        return self.data[pos]


class FoldEnsemble:
    """Moyenne des probabilités des modèles de la validation croisée (model.pkl des expériences)."""

    def __init__(self, models: list):
        self.models = models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def _stratified(clips: dict[str, set[str]], ratios: dict, seed: int) -> dict[str, object]:
    """Répartit des chaînes entre des parties (ratios : part visée de chacune) en gardant chaque
    caractéristique au plus près des proportions visées : stratification itérative, chaînes aux
    caractéristiques les plus rares d'abord."""
    total = Counter(t for traits in clips.values() for t in traits)
    rng = np.random.default_rng(seed)
    order = list(clips)
    rng.shuffle(order)
    order.sort(key=lambda c: min((total[t] for t in clips[c]), default=10 ** 9))
    n = len(clips)
    # volumes visés en chaînes, au plus fort reste : leur somme vaut toujours n
    raw = {s: r * n for s, r in ratios.items()}
    cap = {s: int(v) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: raw[s] - cap[s], reverse=True)[:n - sum(cap.values())]:
        cap[s] += 1
    got, got_t, assign = Counter(), {s: Counter() for s in ratios}, {}
    for c in order:
        def deficit(s):
            # Déficits relatifs (rapportés à la cible) : en absolu, un cas rare (vu une seule
            # fois) partirait toujours dans la partie la plus grosse.
            trait = sum(1 - got_t[s][t] / (ratios[s] * total[t]) for t in clips[c])
            return trait, 1 - got[s] / (ratios[s] * n), rng.random()
        s = max([s for s in ratios if got[s] < cap[s]], key=deficit)
        assign[c] = s
        got[s] += 1
        got_t[s].update(clips[c])
    return assign


def _orient(meta: dict) -> str:
    return "portrait" if meta["video"]["height"] > meta["video"]["width"] else "paysage"


def _video_traits(meta: dict, empty: bool, keys: list[str]) -> set[str]:
    """Caractéristiques d'une vidéo selon les clés de stratification (STRATIFY_KEYS)."""
    out = set()
    for k in keys:
        if k == "orientation":
            out.add(f"orientation:{_orient(meta)}")
        elif k == "clip_vide" and empty:
            out.add("clip_vide")
    return out


class Lab:
    def __init__(self, data_budget: int = 250, final_test_budget: int = 2, max_per_request: int = 20,
                 annotation_budget_usd: float = 2.0, direct_annotation: bool = False):
        self.data_budget = data_budget
        # Annotation en appels directs (prix plein, résultats immédiats) au lieu des batchs (-50 %,
        # jusqu'à des heures d'attente) : agents --direct-annotation.
        self.direct_annotation = direct_annotation
        self.max_per_request = max_per_request
        self.generated = 0
        self.annotation_budget_usd = annotation_budget_usd
        self.annotation_spent = 0.0  # coût réel de l'annotation de la session (batchs récupérés ou appels directs)
        self.final_tests_left = final_test_budget
        self._rows: list[dict] | None = None
        self._ann_cache: dict[str, dict] = {}
        self._folds_cache: dict[str, dict[str, int]] = {}
        self._test_chains_cache: dict[str, int] = {}
        self._empty_cache: tuple[list, set[str]] | None = None
        EXP_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ données
    @property
    def rows(self) -> list[dict]:
        if self._rows is None:
            if not (FRAMES_DIR / "index.jsonl").exists():
                build()
            self._rows = load_index()
        return self._rows

    def _split_map(self, split_id: str) -> dict[str, str]:
        """sample_id -> dev/test. Les clips exclus à la création du split, ou collectés après,
        n'y figurent pas. Une exclusion postérieure ne change pas un split existant : son val et
        son test restent ceux sur lesquels ses expériences ont été mesurées."""
        if split_id == LEGACY_SPLIT:
            raise ValueError("le split 'hash' n'est plus utilisable (il n'était pas figé) : crée un split "
                             "avec make_split et ré-entraîne dessus")
        path = SPLITS_DIR / f"{split_id}.json"
        if not path.exists():
            raise ValueError(f"split inconnu : {split_id} (splits existants dans list_experiments ; sinon make_split)")
        sp = json.loads(path.read_text(encoding="utf-8"))
        excluded = set(sp["excluded"])
        part = {"train": "dev", "val": "dev"}  # anciens splits train/val/test
        return {r["sample_id"]: part.get(sp["clips"][r["source"]], sp["clips"][r["source"]]) for r in self.rows
                if r["source"] in sp["clips"] and r["sample_id"] not in excluded}

    def _channel_traits(self, sids, stratify: list[str]) -> dict[str, set[str]]:
        """Chaîne -> caractéristiques de ses vidéos (clés de stratification)."""
        clips: dict[str, set[str]] = {}
        empty = self._empty_samples()
        for sid in sorted(sids):
            ann = self._ann(sid)
            clips.setdefault(source_key(ann), set()).update(_video_traits(ann, sid in empty, stratify))
        return clips

    def _folds(self, split_id: str) -> dict[str, int]:
        """Chaîne de dev -> pli de validation croisée, stratifié comme le split (mêmes clés, même
        graine). Enregistré dans le split par make_split : recalculé, il bougerait avec les
        étiquettes (clip_vide) ou le code de _stratified, et le val avec lui. Les splits d'avant
        l'enregistrement (split_001 à split_004) n'en ont pas : recalculés."""
        if split_id not in self._folds_cache:
            sp = json.loads((SPLITS_DIR / f"{split_id}.json").read_text(encoding="utf-8"))
            if "folds" in sp:
                self._folds_cache[split_id] = sp["folds"]
                return sp["folds"]
            params = sp["params"]
            dev = {s for s, v in self._split_map(split_id).items() if v == "dev"}
            clips = self._channel_traits(dev, params.get("stratify", list(STRATIFY_KEYS)))
            k = min(FOLDS, len(clips))
            if k < 2:
                raise ValueError("il faut au moins 2 chaînes en dev pour la validation croisée")
            self._folds_cache[split_id] = _stratified(clips, {f: 1 / k for f in range(k)}, params.get("seed", 0))
        return self._folds_cache[split_id]

    def _excluded(self) -> dict[str, str]:
        return json.loads(EXCLUDED.read_text(encoding="utf-8")) if EXCLUDED.exists() else {}

    def _idx(self, split: str, split_id: str) -> np.ndarray:
        """Images étiquetées du split (celles sur lesquelles on entraîne et on mesure) : les images
        sans accord des annotateurs n'y sont jamais."""
        m = self._split_map(split_id)
        return np.array([i for i, r in enumerate(self.rows)
                         if m.get(r["sample_id"]) == split and r["label_any"] is not None], dtype=int)

    def _y(self) -> np.ndarray:
        # 0 pour les images sans étiquette : elles ne sont jamais sélectionnées par _idx.
        return np.array([r["label_any"] or 0 for r in self.rows])

    def _positive_samples(self) -> set[str]:
        """Vidéos ayant au moins une image avec ajout (accord des annotateurs)."""
        return {r["sample_id"] for r in self.rows if r["label_any"] == 1}

    def _empty_samples(self) -> set[str]:
        """Vidéos sans aucun ajout : des images étiquetées, toutes brutes (une vidéo pas encore
        annotée n'en fait pas partie). Gardé tant que l'index ne change pas : _balance_groups
        l'appelle pour chaque image, et le recalcul parcourait tout l'index à chaque fois (run
        020 : environ 50 min par train_model, dont l'essentiel ici)."""
        rows = self.rows
        if self._empty_cache is None or self._empty_cache[0] is not rows:
            labeled = {r["sample_id"] for r in rows if r["label_any"] is not None}
            self._empty_cache = (rows, labeled - self._positive_samples())
        return self._empty_cache[1]

    def _ann(self, sample_id: str) -> dict:
        """meta.json de la vidéo (titre, chaîne, dimensions, requête)."""
        if sample_id not in self._ann_cache:
            self._ann_cache[sample_id] = youtube.load_meta(SAMPLES_DIR / sample_id)
        return self._ann_cache[sample_id]

    def _X(self, features: list[str]) -> tuple["LabeledRows", float]:
        """Embeddings des seules images étiquetées (les seules que _idx sélectionne), indexés par
        numéro de ligne de self.rows. Une matrice sur tout l'index, à 0 hors étiquettes, faisait
        4x la taille utile : run 025 tué à 5 Go (MemoryMax) avec 11 embeddings."""
        mats, ms = [], 0.0
        labeled = np.array([i for i, r in enumerate(self.rows) if r["label_any"] is not None], dtype=int)
        for spec in features:
            ext, _, view = spec.partition(":")
            feats = get_features([self.rows[i] for i in labeled], ext, view or "full")
            if feats is not None:
                mats.append(np.asarray(feats, dtype=np.float32))
            ms += extraction_cost(ext, view or "full") or 0.0
        X = np.concatenate(mats, axis=1) if mats else np.zeros((len(labeled), 0), dtype=np.float32)
        del mats
        return LabeledRows(X, labeled, len(self.rows)), ms

    # ------------------------------------------------------------------ outils
    def _composition(self, sids: set[str], with_positives: bool = True) -> dict:
        """Composition d'un ensemble de vidéos : volumes, orientation, accord des annotateurs."""
        rows = [r for r in self.rows if r["sample_id"] in sids]
        labeled = [r for r in rows if r["label_any"] is not None]
        anns = [self._ann(s) for s in sorted(sids)]
        chains = {r["source"] for r in rows}
        out = {"chaines_distinctes": len(chains), "videos": len(sids),
               "moyenne_videos_par_chaine": round(len(sids) / len(chains), 2) if chains else 0,
               "images_extraites": len(rows), "images_etiquetees": len(labeled),
               # annotées mais sans étiquette retenue (désaccord des annotateurs)
               "images_sans_etiquette": sum(r["label_any"] is None and r.get("annotee", True) for r in rows),
               # jamais envoyées aux annotateurs : quasi identiques à la précédente annotée, ou
               # annotation en attente
               "images_non_annotees": sum(not r.get("annotee", True) for r in rows),
               "videos_sans_ajout": len(sids & self._empty_samples()),
               "videos_par_orientation": dict(Counter(_orient(a) for a in anns))}
        if with_positives:
            out["pourcentage_images_avec_ajout"] = (
                round(100 * sum(r["label_any"] for r in labeled) / len(labeled), 1) if labeled else None)
        return out

    def latest_split(self) -> str | None:
        """Le split le plus récent, celui sur lequel entraîner (None s'il n'y en a pas)."""
        splits = sorted(p.stem for p in SPLITS_DIR.glob("split_*.json"))
        return splits[-1] if splits else None

    def _split_summaries(self) -> list[dict]:
        out = []
        all_sids = {r["sample_id"] for r in self.rows}
        for sid in sorted(p.stem for p in SPLITS_DIR.glob("split_*.json")):
            m = self._split_map(sid)
            sp = json.loads((SPLITS_DIR / f"{sid}.json").read_text(encoding="utf-8"))
            out.append({"split_id": sid, **{s: sum(v == s for v in m.values()) for s in SPLITS},
                        "note": sp["note"], "date": sp["date"], "params": sp["params"],
                        "clips_exclus": len(sp["excluded"]),
                        # vidéos collectées après la création du split
                        "clips_hors_split": len(all_sids - set(m) - set(sp["excluded"]))})
        return out

    def _video_cost_usd(self, rows: list[dict]) -> float | None:
        """Coût estimé de l'annotation d'une nouvelle vidéo : coût réel moyen d'une image par les trois
        annotateurs (batchs déjà récupérés, prix plein en annotation directe) x images d'une vidéo
        moyenne du dataset, moins les quasi-doublons sautés. None tant qu'un annotateur n'a pas de
        coût mesuré."""
        state = batch._load()
        per_image = [batch._measured(m, state) for m in annotators.ANNOTATORS]
        if not rows or None in per_image:
            return None
        frames_per_video = len(rows) / len({r["sample_id"] for r in rows})
        factor = 1 / batch.DISCOUNT if self.direct_annotation else 1.0
        return sum(per_image) * factor * frames_per_video * (1 - SKIPPED_SHARE)

    def dataset_info(self, split_id: str | None = None) -> dict:
        rows = self.rows
        left_usd = self.annotation_budget_usd - self.annotation_spent
        budgets = {"videos_collectables_restantes": self.data_budget - self.generated,
                   "budget_annotation_restant_usd": round(left_usd, 3)}
        per_video = self._video_cost_usd(rows)
        if per_video:
            budgets["videos_annotables_estimees"] = min(int(left_usd / per_video), budgets["videos_collectables_restantes"])
        budgets["final_tests_restants"] = self.final_tests_left
        if not rows:
            return {"dataset_vide": True, "budgets": budgets,
                    "message": "Aucune vidéo : le dataset est à constituer (generate_data)."}
        sids = sorted({r["sample_id"] for r in rows})
        info = {"dataset_total_disponible": self._composition(set(sids)), "budgets": budgets}
        if split_id:
            m = self._split_map(split_id)
            excluded = json.loads((SPLITS_DIR / f"{split_id}.json").read_text(encoding="utf-8"))["excluded"]
            info["composition_du_dernier_split"] = {
                "split_id": split_id,
                **{s: self._composition({k for k, v in m.items() if v == s}, with_positives=s != "test")
                   for s in SPLITS},
                # vidéos collectées depuis : un nouveau split les ferait compter
                "videos_hors_split": len(set(sids) - set(m) - set(excluded))}
        # Plusieurs vidéos d'une même chaîne partagent son habillage : elles diversifient peu.
        chains = Counter(source_key(self._ann(s)) for s in sids)
        several = {c: n for c, n in chains.most_common(TOP_GROUPS) if n >= 2}
        rest = [n for _, n in chains.most_common()[TOP_GROUPS:] if n >= 2]
        if rest:  # au plus TOP_GROUPS chaînes, le reste en un total
            several["autres"] = f"{len(rest)} autres chaînes, {sum(rest)} vidéos"
        if several:
            info["dataset_total_disponible"]["chaines_avec_plusieurs_videos"] = several
        return _jsonable(info)

    def generate_data(self, n: int, reason: str, query: str, orientation: str | None = None) -> dict:
        """Collecte n nouvelles vraies vidéos YouTube pour la requête, puis fait annoter leurs images
        par les trois LLM (appels payants, plafonnés par annotation_budget_usd) : en mode batch, en
        attendant les résultats sans limite (envoi et récupération dans le même appel : un batch que
        l'appel n'a pas récupéré, s'il est interrompu, est oublié), sauf les annotateurs de
        annotators.DIRECT_ANNOTATORS, toujours en appels directs ; tout en appels directs si
        direct_annotation. Une vidéo non annotée (budget épuisé) reste sans étiquette, jamais reprise."""
        n = min(int(n), self.max_per_request, self.data_budget - self.generated)
        if n <= 0:
            return {"error": "budget de vidéos épuisé pour cette session"}
        if not (query or "").strip():
            raise ValueError("query obligatoire : la recherche YouTube")
        if self.annotation_spent >= self.annotation_budget_usd:
            return {"error": "budget d'annotation épuisé pour cette session"}
        # Une vidéo par chaîne absente du dataset : deux vidéos d'une chaîne partagent son habillage,
        # et le split se fait par chaîne.
        avoid = youtube.channels()
        try:
            got = youtube.collect(query, n, orientation, avoid_channels=avoid)
        except youtube.NoVideoFound as e:
            got = {"dirs": [], "errors": [str(e)]}
        self.generated += len(got["dirs"])  # seules les vidéos téléchargées consomment le budget
        errors, chosen, est, stopped = list(got["errors"]), [], 0.0, None
        factor = 1 / batch.DISCOUNT if self.direct_annotation else 1.0  # estimations au tarif batch
        for d in got["dirs"]:
            e = batch.estimate([d])["total_usd_estime"] * factor
            if self.annotation_spent + est + e > self.annotation_budget_usd:
                stopped = "budget d'annotation épuisé : vidéos suivantes téléchargées mais laissées sans étiquette"
                break
            chosen.append(d)
            est += e
        before = self.annotation_spent
        failed = 0
        # Appels directs : tous les annotateurs (direct_annotation), sinon ceux dont le batch est trop
        # lent (annotators.DIRECT_ANNOTATORS) ; le batch ci-dessous n'envoie que les votes manquants.
        direct = annotators.ANNOTATORS if self.direct_annotation else annotators.DIRECT_ANNOTATORS
        for d in chosen:
            res = annotate(d, max_usd=self.annotation_budget_usd - self.annotation_spent, models=direct)
            self.annotation_spent += res["cost_usd"]
            if res["arret_budget"]:
                stopped = "budget d'annotation épuisé pendant l'annotation directe"
                break
        # 2e passage pour les requêtes en échec (ex. effort « medium » refusé, renvoyé en « none »)
        for _ in range(2 if chosen and not self.direct_annotation else 0):
            try:
                sent = batch.submit(float("inf"), samples=chosen)["envoyes"]
            except Exception as e:
                errors.append(f"envoi en batch en échec : {str(e)[:160]}")
                break
            if not sent:
                break
            jobs = {s["job_id"] for s in sent}
            batch.wait(jobs)
            res = batch.collect(job_ids=jobs)
            self.annotation_spent += res["cost_usd"]
            failed = sum(res["echecs"].values())
            if not failed:
                break
        build()
        self._rows = None
        sids = {d.name for d in chosen}
        mine = [r for r in self.rows if r["sample_id"] in sids]
        return _jsonable({
            "videos_collectees": len(got["dirs"]),  # autant de chaînes nouvelles (une vidéo chacune)
            "echecs": len(errors), "exemples_erreurs": errors[:3], "arret": stopped,
            # images des vidéos collectées : toutes, celles retenues pour l'annotation (sans les
            # quasi identiques, frames.to_annotate) et celles annotées avec accord des trois LLM
            "total_images_extraites_videos": len(mine),
            "total_images_pour_annotation": sum(len(to_annotate(d)) for d in got["dirs"]),
            "total_images_annotees": sum(r["label_any"] is not None for r in mine),
            "requetes_en_echec": failed,
            "cout_annotation_usd": round(self.annotation_spent - before, 4),
            "nouveau_total_global_images_etiquetees": sum(r["label_any"] is not None for r in self.rows),
            "videos_collectables_restantes": self.data_budget - self.generated,
            "budget_annotation_restant_usd": round(self.annotation_budget_usd - self.annotation_spent, 3)})

    def data_requested_history(self) -> dict:
        """Requêtes YouTube qui ont amené des vidéos au dataset, avec leur nombre par orientation (lu
        dans les meta.json des vidéos), de la plus productive à la moins productive ; au plus
        HISTORY_QUERIES, le reste en un total. Les collectes sans résultat n'y sont pas : leur échec
        peut venir d'un blocage YouTube autant que de la requête."""
        tally: dict[str, Counter] = defaultdict(Counter)
        for sid in {r["sample_id"] for r in self.rows}:
            meta = self._ann(sid)
            tally[meta.get("query") or "(requête non notée)"][_orient(meta)] += 1
        ranked = sorted(tally.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
        out = {q: ", ".join(f"{n} {o}" for o, n in sorted(c.items())) for q, c in ranked[:HISTORY_QUERIES]}
        rest = ranked[HISTORY_QUERIES:]
        if rest:
            out["(autres requêtes)"] = f"{len(rest)} requêtes, {sum(sum(c.values()) for _, c in rest)} vidéos"
        return out

    def exclude_clips(self, clips: dict) -> dict:
        """Retire des clips des splits créés ensuite : {sample_id: raison}. Les splits existants ne
        changent pas, pour qu'une exclusion ne puisse pas rendre un val déjà mesuré plus facile.
        Réservé aux défauts constatés sur le clip lui-même (vidéo cassée, doublon), jamais aux
        erreurs d'un modèle. Les fichiers ne sont pas supprimés."""
        all_sids = {r["sample_id"] for r in self.rows}
        if not clips:
            raise ValueError("aucun clip à exclure")
        if set(clips) - all_sids:
            raise ValueError(f"clips inconnus : {sorted(set(clips) - all_sids)}")
        if any(not str(v).strip() for v in clips.values()):
            raise ValueError("chaque exclusion doit avoir une raison")
        excluded = {**self._excluded(), **{k: str(v) for k, v in clips.items()}}
        write_atomic(EXCLUDED, json.dumps(excluded, ensure_ascii=False, indent=2))
        return {"clips_exclus": excluded,
                "effet": "appliqué aux splits créés à partir de maintenant (make_split) ; splits existants inchangés"}

    def make_split(self, note: str, test_ratio: float = 0.2, stratify: list[str] | None = None,
                   seed: int = 0) -> dict:
        """Crée un split dev/test figé. Toutes les vidéos d'une même chaîne YouTube vont du même
        côté. Chaque split tire son propre test, test_ratio des chaînes (au plus TEST_MAX_CHAINS),
        en répartissant chaque caractéristique de stratify (orientation, clip vide) au plus près ;
        ses expériences sont testées sur ce test, qui ne se compare pas à celui d'un autre split.
        stratify (par défaut toutes, [] pour un tirage purement aléatoire) et seed fixent aussi
        les plis de dev (_folds)."""
        # Par défaut tout est stratifié : sans ça, un cas rare peut tomber entièrement en test et
        # jamais en dev (run 004).
        stratify = list(STRATIFY_KEYS) if stratify is None else list(stratify)
        exclude = self._excluded()
        if not note.strip():
            raise ValueError("note obligatoire : pourquoi ce split")
        if not 0.05 <= test_ratio <= 0.4:
            raise ValueError("test_ratio doit être entre 0.05 et 0.4")
        if set(stratify) - set(STRATIFY_KEYS):
            raise ValueError(f"stratify parmi {STRATIFY_KEYS}")
        clips = self._channel_traits({r["sample_id"] for r in self.rows} - set(exclude), stratify)
        if len(clips) < 3:
            raise ValueError("il faut au moins 3 chaînes YouTube pour faire un split")
        ratio = min(test_ratio, TEST_MAX_CHAINS / len(clips))
        assign = _stratified(clips, {"dev": 1 - ratio, "test": ratio}, seed)
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        split_id = f"split_{len(list(SPLITS_DIR.glob('split_*.json'))) + 1:03d}"
        params = {"test_ratio": test_ratio, "stratify": stratify, "seed": seed}
        path = SPLITS_DIR / f"{split_id}.json"
        sp = {"split_id": split_id, "note": note, "date": time.strftime("%Y-%m-%d %H:%M"), "params": params,
              "excluded": exclude, "clips": assign}
        write_atomic(path, json.dumps(sp, ensure_ascii=False, indent=2))
        # plis figés avec le split (voir _folds), calculés sur le fichier qui vient d'être écrit
        sp["folds"] = self._folds(split_id)
        write_atomic(path, json.dumps(sp, ensure_ascii=False, indent=2))
        m = self._split_map(split_id)
        return _jsonable({"split_id": split_id, "params": params, "clips_exclus": len(exclude),
                          "plis_de_validation": len(set(self._folds(split_id).values())),
                          **{s: self._composition({k for k, v in m.items() if v == s},
                                                  with_positives=s != "test") for s in SPLITS}})

    def production(self) -> dict | None:
        path = EXP_DIR / "production.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def production_status(self) -> dict | None:
        """Le modèle en production et les données dont il est issu : vidéos de son split contre
        vidéos disponibles aujourd'hui (hors exclues). Moins de vidéos dans son split : un modèle
        entraîné sur tout le dataset actuel reste à faire."""
        prod = self.production()
        if not prod:
            return None
        split = next((sp for sp in self._split_summaries() if sp["split_id"] == prod.get("split_id")), None)
        available = len({r["sample_id"] for r in self.rows} - set(self._excluded()))
        return {"experiment_id": prod["experiment_id"], "split_id": prod.get("split_id"),
                "test": (prod.get("test") or {}).get("statut"),
                "videos_du_split": sum(split[k] for k in SPLITS) if split else None,
                "videos_disponibles": available}

    def train_model(self, embeddings: list[str], model: str, params: dict | None = None, balance: dict | None = None,
                    weights: dict | None = None, note: str = "") -> dict:
        # Toujours sur le split le plus récent : un ancien split ne se compare plus à rien de récent.
        split_id = self.latest_split()
        if not split_id:
            raise ValueError("aucun split : crée-en un avec make_split avant d'entraîner")
        if model not in MODELS:
            raise ValueError(f"model dans {MODELS}")
        if not embeddings or len(set(embeddings)) < len(embeddings):
            raise ValueError("embeddings : au moins un, chacun une seule fois")
        dev = self._idx("dev", split_id)
        if not len(dev):
            raise ValueError("dev vide : il faut d'abord collecter des données")
        params = params or {}
        folds = self._folds(split_id)
        fold = np.array([folds[self.rows[i]["source"]] for i in dev])
        groups = np.array([self.rows[i]["source"] for i in dev])
        X, feat_ms = self._X(embeddings)
        y = self._y()

        t0 = time.time()
        probs, fold_models = np.zeros(len(dev)), []
        for k in np.unique(fold):  # chaque pli prédit par un modèle qui n'a vu aucune de ses chaînes
            out = fold == k
            fold_models.append(self._fit(model, params, dev[~out], balance, weights, X, y)[0])
            probs[out] = fold_models[-1].predict_proba(X[dev[out]])[:, 1]
        # Modèle gardé : la moyenne des modèles de plis, dont les scores sont ceux sur lesquels le
        # seuil est choisi. Un modèle réentraîné sur tout dev est plus confiant : au même seuil, il
        # perdait 4 à 9 points de précision sur les chaînes cachées du banc b1.
        clf = FoldEnsemble(fold_models)
        balance_report = self._balance(dev, balance or {})[1]
        weighted = bool(weights)
        train_s = time.time() - t0
        t0 = time.time()
        clf.predict_proba(X[dev])
        pred_ms = 1000 * (time.time() - t0) / len(dev)

        exp_id = self._next_exp_id()
        result = {
            "val": metrics.summary(y[dev], probs, groups, self._test_chains(split_id)), "train_s": train_s,
            "ms_par_frame": feat_ms + pred_ms, "n_features": X.shape[1],
            "n_dev": len(dev), "plis": len(np.unique(fold)), "data": self._data_snapshot(split_id),
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        if balance_report:
            result["reequilibrage_train"] = balance_report
        if weighted:
            result["ponderation_train"] = {g: f"poids {float(w):g}" for g, w in sorted(weights.items())}
        # Clé "features" gardée dans config.json : l'interface et final_test la relisent.
        config = {"split_id": split_id, "features": embeddings, "model": model,
                  "params": params, "balance": balance or {}, "weights": weights or {}, "note": note}
        d = EXP_DIR / exp_id
        d.mkdir(parents=True)
        sync_dir(EXP_DIR)
        write_atomic(d / "config.json", json.dumps(config, indent=2))
        # Prédictions val (hors pli) indexées par frame : restent valides si le dataset grossit
        # ensuite ; groups (chaîne de chaque image) sert au seuil prudent.
        write_atomic(d / "val_preds.json", json.dumps(
            {"frames": [self.rows[i]["frame"] for i in dev], "y": y[dev].tolist(),
             "probs": [round(float(p), 5) for p in probs], "groups": groups.tolist()}))
        write_atomic(d / "model.pkl", pickle.dumps(clf))
        # En dernier : list_experiments ne retient que les dossiers qui ont metrics.json, donc une
        # expérience interrompue n'apparaît pas sans son modèle.
        write_atomic(d / "metrics.json", json.dumps(_jsonable(result), indent=2))
        return self.train_summary(exp_id)

    def train_summary(self, experiment_id: str) -> dict:
        """Réponse de train_model, relue dans metrics.json : format court, la config est celle de
        l'appel et le détail reste dans le fichier."""
        result = json.loads((EXP_DIR / experiment_id / "metrics.json").read_text(encoding="utf-8"))
        short = {"val": _compact_val(result["val"])}
        for k in ("reequilibrage_train", "ponderation_train"):
            if k in result:
                short[k] = result[k]
        return _jsonable({"experiment_id": experiment_id, **short})

    def analyze_errors(self, experiment_id: str, precision: float = metrics.DEFAULT_PRECISION) -> dict:
        frames, y, probs, op = self._val_at(experiment_id, precision)
        pred = probs >= op["threshold"]
        by_frame = {r["frame"]: r for r in self.rows}
        fn_rows = [by_frame[f] for f, yi, pi in zip(frames, y, pred) if yi and not pi]
        tp_rows = [by_frame[f] for f, yi, pi in zip(frames, y, pred) if yi and pi]
        fp_rows = [by_frame[f] for f, yi, pi in zip(frames, y, pred) if not yi and pi]

        def missed(key):
            tot, miss = Counter(key(r) for r in tp_rows + fn_rows), Counter(key(r) for r in fn_rows)
            return {k: f"{miss[k]}/{tot[k]} ratées" for k in sorted(tot)}

        # Seulement l'orientation : ce que le modèle rate et ce qui le trompe, l'outil le dit par
        # l'analyse visuelle (agents.common.analyze_errors_tool), plus parlante que des comptes par
        # requête ou par chaîne. Pas d'images : l'agent les voit avec show_examples.
        return _jsonable({
            "experiment_id": experiment_id, "precision_visee": precision, **op,
            "positifs_rates_par_orientation": missed(self._orientation),
            "faux_positifs_par_orientation": dict(Counter(self._orientation(r) for r in fp_rows)),
        })

    def show_examples(self, experiment_id: str, kind: str = "missed", n: int = 4,
                      precision: float = metrics.DEFAULT_PRECISION, order: str = "franches",
                      start: int = 0, query: str | None = None, channel: str | None = None,
                      video: str | None = None) -> list[dict]:
        """Renvoie des blocs image (frames val) pour inspection visuelle, une vidéo à tour de rôle
        pour ne pas montrer n fois la même : les erreurs les plus franches (order="franches" :
        ajouts ratés les moins bien notés, fausses alertes les mieux notées), souvent des erreurs
        d'étiquette, ou dans un ordre au hasard (order="hasard", toujours le même), plus
        représentatif de l'ensemble des erreurs. start : rang du premier exemple, pour voir les
        suivants. query, channel, video : seulement les erreurs de cette requête YouTube, chaîne
        ou vidéo (sample_id), telles qu'analyze_errors les nomme."""
        if order not in ("franches", "hasard"):
            raise ValueError("order doit être 'franches' ou 'hasard'")
        frames, y, probs, op = self._val_at(experiment_id, precision)
        pred = probs >= op["threshold"]
        want = (y == 1) & ~pred if kind == "missed" else (y == 0) & pred
        by_frame = {r["frame"]: r for r in self.rows}
        label = "ajout raté" if kind == "missed" else "faux positif"
        filters = {k: v.strip() for k, v in (("query", query), ("channel", channel)) if v}
        idx = [i for i in np.flatnonzero(want)
               if all((self._ann(by_frame[frames[i]]["sample_id"]).get(k) or "").strip() == v
                      for k, v in filters.items())
               and (not video or by_frame[frames[i]]["sample_id"] == video)]
        if not idx and (filters or video):
            return [{"type": "text", "text": "aucune erreur de ce type pour ce filtre (" + ", ".join(
                f"{k}={v}" for k, v in {**filters, "video": video}.items() if v) + ")"}]
        if order == "franches":
            idx.sort(key=lambda i: probs[i] if kind == "missed" else -probs[i])
        else:
            np.random.default_rng(0).shuffle(idx)
        ranked = _round_robin(idx, lambda i: by_frame[frames[i]]["sample_id"])
        start = max(0, int(start))
        shown = ranked[start:start + n]
        if not shown:
            return [{"type": "text", "text": f"aucun exemple à partir du rang {start} ({len(ranked)} erreurs)"}]
        blocks: list[dict] = [{"type": "text", "text": f"exemples {start + 1} à {start + len(shown)} sur "
                                                       f"{len(ranked)} {'ajouts ratés' if kind == 'missed' else 'faux positifs'} (ordre : {order})"}]
        for i in shown:
            r = by_frame[frames[i]]
            ann = self._ann(r["sample_id"])
            blocks += _image_blocks(FRAMES_DIR / r["frame"], f"{label} {r['sample_id']} ({ann['channel']}, "
                                                             f"requête « {ann.get('query')} ») t={r['t']} "
                                                             f"p={probs[i]:.3f} (seuil {op['threshold']:.3f})")
        return blocks

    def describe_errors(self, experiment_id: str, precision: float = metrics.DEFAULT_PRECISION,
                        max_usd: float = 1.0) -> dict:
        """Toutes les erreurs val de l'expérience (ajouts ratés et fausses alertes), chacune avec la
        description de ce qu'on voit sur l'image (describe : batch, mis en cache, refusé au-delà de
        max_usd) : de quoi en tirer les points communs."""
        frames, y, probs, op = self._val_at(experiment_id, precision)
        pred = probs >= op["threshold"]
        errors = [i for i in range(len(frames)) if bool(y[i]) != bool(pred[i])]
        res = describe.describe([frames[i] for i in errors], max_usd)
        by_frame = {r["frame"]: r for r in self.rows}
        items = []
        for i in errors:
            r = by_frame[frames[i]]
            ann = self._ann(r["sample_id"])
            items.append({"erreur": "ajout raté" if y[i] else "fausse alerte", "video": r["sample_id"],
                          "chaine": ann["channel"], "requete": ann.get("query"), "t": r["t"],
                          "p": round(float(probs[i]), 3), "description": res["descriptions"].get(frames[i])})
        return {"erreurs": items, "cost_usd": res["cost_usd"], "nouvelles_descriptions": res["nouvelles"],
                "echecs": res["echecs"]}

    def list_experiments(self) -> list[dict]:
        """Tout ce qu'on sait de chaque expérience : config, métriques val, volume de données,
        test final et statut de production."""
        prod = self.production() or {}
        latest = self.latest_split()
        key = f"rappel_a_precision_{metrics.DEFAULT_PRECISION}"
        out = []
        for d in sorted(p for p in EXP_DIR.glob("exp_*") if p.is_dir()):
            if not (d / "metrics.json").exists():
                continue
            try:
                e = {"id": d.name,
                     **json.loads((d / "config.json").read_text(encoding="utf-8")),
                     **json.loads((d / "metrics.json").read_text(encoding="utf-8"))}
                if (d / "final_test.json").exists():
                    e["final_test"] = json.loads((d / "final_test.json").read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError):
                # Fichiers illisibles (run 009 : arrêt brutal de la machine, fichiers remplis de zéros) :
                # l'expérience est ignorée plutôt que de casser list_experiments et la fin de run.
                print(f"[lab] {d.name} ignorée : fichiers illisibles", file=sys.stderr)
                continue
            # Split récent : seuil d'un ancien réglage recalculé une fois (réécrit par _val_at), pour
            # que les agents comparent les expériences au même point que celui que final_test teste.
            if (e.get("split_id") == latest and
                    self._prudent_stale(e["val"].get(key), latest, e["val"].get("ap_par_chaine") is not None)):
                try:
                    self._val_at(d.name, metrics.DEFAULT_PRECISION)
                except ValueError:
                    pass
                e["val"] = json.loads((d / "metrics.json").read_text(encoding="utf-8"))["val"]
            e["en_production"] = prod.get("experiment_id") == d.name
            out.append(e)
        return out

    def experiments_overview(self, brief: bool = False) -> dict:
        """Ce que renvoie l'outil list_experiments : le split le plus récent (celui sur lequel
        entraîner) avec ses expériences, et une ligne par ancien split ayant des expériences (elles
        ne se comparent pas à celles du récent). Sur le split récent, seules celles qui comptent
        pour décider tiennent chacune en une ligne (les LISTED_BEST meilleures, celles passées au
        test final ou en production, les LISTED_RECENT dernières), les autres en un total, pour que
        la réponse ne grandisse pas avec le nombre d'expériences. En tête, le modèle en production ;
        sur le split récent, candidat_a_tester, la prochaine expérience à passer au test final
        (candidate_to_test). brief (orchestrateur) : du split récent, les vidéos hors split, la
        note, candidat_a_tester et les BRIEF_BEST meilleures en une ligne."""
        exps = self.list_experiments()
        summaries = self._split_summaries()
        latest = summaries[-1]["split_id"] if summaries else None
        y = self._y()
        out = []
        for sp in summaries:
            sid = sp["split_id"]
            mine = [e for e in exps if e.get("split_id") == sid]
            if sid != latest:
                if mine:
                    out.append(_split_line(
                        sid, mine, lambda i: self._test_status(i, metrics.DEFAULT_PRECISION)["statut"]))
                continue
            note = sp["note"]
            note = note if len(note) <= 120 else note[:120].rstrip() + "…"
            ranked = sorted(mine, key=_score, reverse=True)
            if brief:
                entry = {"split_id": sid, "clips_hors_split": sp["clips_hors_split"], "note": note,
                         "candidat_a_tester": self.candidate_to_test(sid), "experiences": len(mine),
                         "meilleures": {e["id"]: _one_line(e) for e in ranked[:BRIEF_BEST]}}
                blocked = self.retest_blocked(sid)
                if blocked:
                    entry["test_bloque"] = blocked
                out.append(entry)
                continue
            dev = self._idx("dev", sid)
            # Chaînes par partie, recomptées sur les clips réellement assignés : chaines_shared
            # (chaînes présentes dans plusieurs parties) doit rester à 0, sinon test fuit dans dev.
            m = self._split_map(sid)
            parts: dict[str, set[str]] = {}
            for r in self.rows:
                if r["sample_id"] in m:
                    parts.setdefault(r["source"], set()).add(m[r["sample_id"]])
            entry = {"split_id": sid,
                     **{f"chaines_{s}": sum(ps == {s} for ps in parts.values()) for s in SPLITS},
                     **{f"clips_{s}": sp[s] for s in SPLITS},
                     "images_dev": len(dev), "images_dev_positives": int(y[dev].sum()),
                     "clips_hors_split": sp["clips_hors_split"]}
            if sp["clips_exclus"]:
                entry["clips_exclus"] = sp["clips_exclus"]
            shared = sum(len(ps) > 1 for ps in parts.values())
            if shared:  # impossible par construction : test fuirait dans dev
                entry["anomalie_chaines_dans_dev_et_test"] = shared
            entry["note"] = note
            entry["candidat_a_tester"] = self.candidate_to_test(sid)
            blocked = self.retest_blocked(sid)
            if blocked:
                entry["test_bloque"] = blocked
            entry["experiences"] = len(mine)
            listed = {}
            for name, group in (("meilleures", ranked[:LISTED_BEST]),
                                ("testees_ou_en_production",
                                 [e for e in mine if e.get("en_production") or "final_test" in e]),
                                ("dernieres", mine[-LISTED_RECENT:])):
                group = {e["id"]: _one_line(e) for e in group if e["id"] not in listed}
                if group:
                    entry[name] = group
                    listed.update(group)
            rest = [_score(e) for e in mine if e["id"] not in listed]
            if rest:
                entry["autres"] = f"{len(rest)} expériences, AP par chaîne de {min(rest)} à {max(rest)}"
            out.append(entry)
        # En tête : le modèle en production, quel que soit son split (la production passe d'abord par
        # un test réussi, d'où l'écart possible avec la meilleure en val).
        prod = next((e for e in exps if e.get("en_production")), None)
        head = ({"en_production": {prod["id"]: f"{prod['split_id']}, " + _one_line(
                     prod, self._test_status(prod["id"], metrics.DEFAULT_PRECISION)["statut"])}}
                if prod else {"en_production": None})
        return {**head, "splits": out}

    def final_test(self, experiment_id: str, precision: float = metrics.DEFAULT_PRECISION) -> dict:
        if self.final_tests_left <= 0:
            return {"error": "plus de final_test disponible dans cette session"}
        _, _, _, op = self._val_at(experiment_id, precision)
        d = EXP_DIR / experiment_id
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        split_id = config.get("split_id", LEGACY_SPLIT)
        blocked = self.retest_blocked(split_id, precision)
        if blocked:
            return {"error": blocked}
        te = self._idx("test", split_id)  # avant de décompter : un split refusé ne coûte pas de test
        self.final_tests_left -= 1
        with open(d / "model.pkl", "rb") as fh:
            clf = pickle.load(fh)
        X, _ = self._X(config["features"])
        y = self._y()[te]
        probs = clf.predict_proba(X[te])[:, 1]
        s = metrics.summary(y, probs)
        result = {"experiment_id": experiment_id, "split_id": split_id,
                  "precision_visee": precision, "seuil_choisi_sur_val": op["threshold"], "val": op,
                  "test": metrics.operating_point(y, probs, op["threshold"]),
                  "test_auc": s["auc"], "test_ap": s["ap"], "final_tests_restants": self.final_tests_left,
                  "contre_production": self._against_production(experiment_id, te, probs, op["threshold"])}
        write_atomic(d / "final_test.json", json.dumps(_jsonable(result), indent=2))
        return self.final_test_summary(experiment_id, precision)

    def _against_production(self, experiment_id: str, te: np.ndarray, probs: np.ndarray,
                            threshold: float) -> dict | None:
        """Le candidat et le modèle en production, chacun à son seuil, sur les chaînes de test que
        la production n'a jamais vues (hors du dev de son split) : de quoi voir si le candidat fait
        mieux qu'elle, sur les mêmes images, même quand ce test est trop dur pour la précision
        visée (run 021 : exp_361, en production, tenait 0,72 de précision sur ces chaînes, les
        trois candidats 0,81 à 0,82, tous échoués à 0,90 ; il est resté en production). None
        sans production, ou avec moins de MIN_CHAINS_VS_PROD chaînes communes."""
        path = EXP_DIR / "production.json"
        if not path.exists():
            return None
        prod = json.loads(path.read_text(encoding="utf-8"))
        pid = prod["experiment_id"]
        if pid == experiment_id:
            return None
        seen = {self.rows[i]["source"] for i in self._idx("dev", prod["split_id"])}
        keep = np.array([k for k, i in enumerate(te) if self.rows[i]["source"] not in seen], dtype=int)
        chains = {self.rows[te[k]]["source"] for k in keep}
        if len(chains) < MIN_CHAINS_VS_PROD:
            return None
        pd = EXP_DIR / pid
        pconfig = json.loads((pd / "config.json").read_text(encoding="utf-8"))
        with open(pd / "model.pkl", "rb") as fh:
            pclf = pickle.load(fh)
        PX, _ = self._X(pconfig["features"])
        y = self._y()[te[keep]]
        pprobs = pclf.predict_proba(PX[te[keep]])[:, 1]
        side = lambda p, t: {**metrics.operating_point(y, p, t), "ap": metrics.summary(y, p)["ap"]}
        return {"production": pid, "production_verifiee": prod["test"].get("statut") == "réussi",
                "chaines": len(chains), "images": len(keep),
                "candidat": side(probs[keep], threshold), "modele_production": side(pprobs, prod["threshold"])}

    def compare_tested_to_production(self, split_id: str) -> dict:
        """Ajoute contre_production aux tests finaux déjà passés sur ce split qui ne l'ont pas
        (faits avant la règle) : relit le test, sans décompter de final_test ni rien choisir."""
        out = {}
        for p in sorted(EXP_DIR.glob("exp_*/final_test.json")):
            ft = json.loads(p.read_text(encoding="utf-8"))
            if ft.get("split_id") != split_id or "contre_production" in ft:
                continue
            eid = p.parent.name
            config = json.loads((p.parent / "config.json").read_text(encoding="utf-8"))
            with open(p.parent / "model.pkl", "rb") as fh:
                clf = pickle.load(fh)
            te = self._idx("test", split_id)
            X, _ = self._X(config["features"])
            ft["contre_production"] = self._against_production(
                eid, te, clf.predict_proba(X[te])[:, 1], ft["seuil_choisi_sur_val"])
            write_atomic(p, json.dumps(_jsonable(ft), indent=2))
            out[eid] = {**self._test_status(eid, ft["precision_visee"]), "contre_production": ft["contre_production"]}
        return _jsonable(out)

    def final_test_summary(self, experiment_id: str, precision: float = metrics.DEFAULT_PRECISION) -> dict:
        """Réponse de final_test, relue dans final_test.json : le verdict en tête, calculé comme pour
        le choix de la production (_test_status). Sans lui, l'Evaluator jugeait lui-même (run 010 :
        « échec relatif » d'un test réussi, parce que le rappel baissait)."""
        result = json.loads((EXP_DIR / experiment_id / "final_test.json").read_text(encoding="utf-8"))
        status = self._test_status(experiment_id, precision)
        status = {k: status[k] for k in ("statut", "motif") if k in status}
        return _jsonable({**status, "regle": "réussi si la précision visée tient sur test, au seuil "
                          "choisi sur val, ou si le candidat fait mieux que le modèle en production (précision "
                          "et AP) sur les chaînes de test que celui-ci n'a jamais vues ; le rappel n'entre pas "
                          "dans le verdict", **result})

    def save_model(self, experiment_id: str, precision: float, note: str,
                   test: dict | None = None) -> dict:
        _, _, _, op = self._val_at(experiment_id, precision)
        config = json.loads((EXP_DIR / experiment_id / "config.json").read_text(encoding="utf-8"))
        split_id = config.get("split_id", LEGACY_SPLIT)
        self._split_map(split_id)  # refuse le split 'hash' : son test n'a jamais été figé
        prod = {"experiment_id": experiment_id, "split_id": split_id,
                "precision_visee": precision, "threshold": op["threshold"],
                "val_recall": op["recall"], "val_precision": op["precision"],
                "test": test or self._test_status(experiment_id, precision), "note": note, "time": time.time()}
        write_atomic(EXP_DIR / "production.json", json.dumps(_jsonable(prod), indent=2))
        return _jsonable(prod)

    def _ranked(self, split_id: str, precision: float = metrics.DEFAULT_PRECISION) -> list[dict]:
        """Expériences d'un split avec ce que regarde la règle de choix (_best_by_rule) et le statut
        de leur test final, triées par AP par chaîne sur val."""
        ranked = []
        for e in self.list_experiments():
            if e.get("split_id") != split_id:
                continue
            try:
                op = self._val_at(e["id"], precision)[3]
            except ValueError:  # précision jamais atteinte sur val
                op = None
            ranked.append({"experiment_id": e["id"], "ap_par_chaine": self._group_ap(e),
                           "val_recall": op["recall"] if op else 0.0,
                           "ap": e["val"]["ap"] or 0.0, "auc": e["val"]["auc"] or 0.0,
                           "ms_par_frame": e["ms_par_frame"], "test": self._test_status(e["id"], precision)})
        ranked.sort(key=lambda r: (-(r["ap_par_chaine"] or 0.0), -r["val_recall"], -r["ap"], -r["auc"],
                                   r["ms_par_frame"]))
        return ranked

    def _annotatable_left(self) -> int:
        """Vidéos qu'on peut encore collecter et faire annoter (videos_annotables_estimees de
        dataset_info ; sans coût mesuré, les vidéos collectables restantes)."""
        left = self.data_budget - self.generated
        per_video = self._video_cost_usd(self.rows) if left > 0 else None
        if per_video:
            left = min(left, int((self.annotation_budget_usd - self.annotation_spent) / per_video))
        return max(left, 0)

    def retest_blocked(self, split_id: str, precision: float = metrics.DEFAULT_PRECISION) -> str | None:
        """Raison de refuser un nouveau test final sur ce split, ou None. Un test y a déjà échoué et
        il reste des vidéos à collecter : un nouveau split, plus gros, vaut mieux qu'un autre test
        d'une variante du même (run 019 : 3 tests sur split_008, tous échoués, 200 vidéos et 1,94 $
        d'annotation laissés)."""
        failed = []
        for p in sorted(EXP_DIR.glob("exp_*/final_test.json")):  # peu de fichiers : pas list_experiments
            try:
                tested = json.loads(p.read_text(encoding="utf-8")).get("split_id")
            except (ValueError, UnicodeDecodeError):
                continue
            if tested == split_id and self._test_status(p.parent.name, precision)["statut"] == "échoué":
                failed.append(p.parent.name)
        left = self._annotatable_left() if failed else 0
        if not left:
            return None
        return (f"test déjà échoué sur {split_id} ({', '.join(failed)}) et {left} vidéos encore "
                "annotables : fais d'abord collecter, puis un nouveau split, avant un autre test final")

    def candidate_to_test(self, split_id: str, precision: float = metrics.DEFAULT_PRECISION) -> str | None:
        """Prochaine expérience à passer au test final : la règle de choix (_best_by_rule) appliquée
        aux non testées du split. Un test échoué la retire, et la suivante prend sa place : les
        tests se font dans l'ordre où select_production choisirait. None une fois un test réussi
        (aucune non testée ne passerait devant), sans non testée, ou tant qu'un nouveau test est
        bloqué (retest_blocked)."""
        ranked = self._ranked(split_id, precision)
        if any(r["test"]["statut"] == "réussi" for r in ranked):
            return None
        if self.retest_blocked(split_id, precision):
            return None
        best = _best_by_rule([r for r in ranked if r["test"]["statut"] == "non testé"])
        return best["experiment_id"] if best else None

    def select_production(self, precision: float = metrics.DEFAULT_PRECISION) -> dict:
        """Choix automatique du modèle de production, fait par le code en fin de run, parmi les
        expériences du plus gros split (le plus de clips ; à égalité, le plus récent) qui ont
        réussi leur test final (précision visée tenue sur test) : parmi celles dont l'AP par
        chaîne sur val est à moins de AP_TIE de la meilleure, celle de meilleur rappel val. À
        défaut de test réussi, même règle sur les non testées, puis sur toutes, marquées comme non
        vérifiées, sauf si la production actuelle a réussi son test : elle est alors conservée
        (production_conservee), même sur un plus petit split. L'AP par chaîne choisit bien mieux
        que le rappel val le modèle qui tient sur des chaînes nouvelles (metrics.py) ; le rappel ne
        départage que des AP équivalentes. Écrit production.json (via save_model) et renvoie le
        classement (par AP par chaîne)."""
        splits = self._split_summaries()
        if not splits:
            return {"error": "aucun split figé : rien à mettre en production"}
        big = max(splits, key=lambda s: (sum(s[k] for k in SPLITS), s["split_id"]))["split_id"]
        ranked = self._ranked(big, precision)
        if not ranked:
            return {"error": f"aucune expérience sur le plus gros split ({big})"}
        order = {"réussi": 0, "non testé": 1, "échoué": 2}
        status = min(order[r["test"]["statut"]] for r in ranked)
        best = _best_by_rule([r for r in ranked if order[r["test"]["statut"]] == status])
        verifie = best["test"]["statut"] == "réussi"
        current_path = EXP_DIR / "production.json"
        current = json.loads(current_path.read_text(encoding="utf-8")) if current_path.exists() else None
        if not verifie and current and current["test"].get("statut") == "réussi":
            # Un modèle vérifié sur son test n'est jamais remplacé par un modèle qui ne l'est pas
            # (run 007 : exp_045, vérifié sur split_002, avait cédé la place à exp_081, non testé).
            return _jsonable({"split_id": big, "production": current, "verifie_sur_test": True,
                              "production_conservee": True, "classement": ranked})
        prod = self.save_model(best["experiment_id"], precision,
                               f"choix automatique de fin de run sur {big} (le plus gros split) : "
                               + (f"meilleur rappel val parmi les AP par chaîne à {AP_TIE} près de la "
                                  "meilleure, chez les modèles qui ont réussi leur test final" if verifie else
                                  "NON VÉRIFIÉ sur test (aucun modèle n'a réussi son test final), meilleur "
                                  f"rappel val parmi les AP par chaîne à {AP_TIE} près de la meilleure "
                                  f"({best['test']['statut']})"),
                               test=best["test"])
        return _jsonable({"split_id": big, "production": prod, "verifie_sur_test": verifie,
                          "production_conservee": False, "classement": ranked})

    def _group_ap(self, e: dict) -> float | None:
        """AP par chaîne sur val d'une expérience ; recalculée depuis val_preds.json pour celles
        entraînées avant qu'elle soit enregistrée (None sans chaînes connues)."""
        if "ap_par_chaine" in e["val"]:
            return e["val"]["ap_par_chaine"]
        preds = json.loads((EXP_DIR / e["id"] / "val_preds.json").read_text(encoding="utf-8"))
        if not preds.get("groups"):
            return None
        return metrics.group_ap(np.array(preds["y"]), np.array(preds["probs"]), preds["groups"])

    def _test_status(self, experiment_id: str, precision: float) -> dict:
        """Résultat du test final d'une expérience pour une précision visée : réussi si la
        précision visée est tenue sur test au seuil choisi sur val, ou si le candidat fait mieux que
        le modèle en production au moment du test, en précision et en AP, sur les chaînes de test
        que celui-ci n'a jamais vues (contre_production) : un test trop dur pour la précision visée
        ne garde plus en production un modèle moins bon."""
        path = EXP_DIR / experiment_id / "final_test.json"
        if not path.exists():
            return {"statut": "non testé"}
        ft = json.loads(path.read_text(encoding="utf-8"))
        if ft["precision_visee"] != precision:
            return {"statut": "non testé", "detail": f"testé à une autre précision visée ({ft['precision_visee']})"}
        t = ft["test"]
        out = {"recall": t["recall"], "precision": t["precision"]}
        if t["precision"] >= precision:
            return {"statut": "réussi", "motif": "précision visée tenue", **out}
        vs = ft.get("contre_production")
        # Battre une production elle-même non vérifiée ne prouve rien.
        if vs and vs["production_verifiee"] and (vs["candidat"]["precision"] >= vs["modele_production"]["precision"]
                   and vs["candidat"]["ap"] >= vs["modele_production"]["ap"]):
            return {"statut": "réussi", "motif": (
                f"meilleur que {vs['production']} (en production au test) sur {vs['chaines']} chaînes "
                f"qu'il n'a jamais vues : précision {vs['candidat']['precision']} contre "
                f"{vs['modele_production']['precision']}, AP {vs['candidat']['ap']} contre "
                f"{vs['modele_production']['ap']}"), **out}
        return {"statut": "échoué", **out}

    # ------------------------------------------------------------------ interne
    @staticmethod
    def _make(model: str, params: dict):
        steps = [StandardScaler()]
        if params.get("pca"):
            steps.append(PCA(n_components=int(params["pca"]), random_state=0))
        if model == "logreg":
            clf = LogisticRegression(C=float(params.get("C", 0.1)), max_iter=3000, class_weight="balanced")
        else:
            # Pas d'early_stopping de sklearn : il met de côté des images au hasard, donc de chaînes
            # vues à l'entraînement ; _fit_mlp arrête sur des chaînes mises de côté.
            clf = MLPClassifier(hidden_layer_sizes=tuple(params.get("hidden", [256])),
                                alpha=float(params.get("alpha", MLP_ALPHA)), random_state=0)
        return make_pipeline(*steps, clf)

    def _fit(self, model: str, params: dict, tr: np.ndarray, balance: dict | None, weights: dict | None,
             X: np.ndarray, y: np.ndarray):
        """Entraîne sur les images tr, rééchantillonnées et pondérées : (classifieur, rapport du
        rééchantillonnage, pondéré ou non)."""
        tr, report = self._balance(tr, balance or {})
        w = self._weights(tr, weights or {})
        clf = self._make(model, params)
        if model == "mlp":
            self._fit_mlp(clf, tr, w, X, y)
        else:
            clf.fit(X[tr], y[tr], **({} if w is None else {f"{clf.steps[-1][0]}__sample_weight": w}))
        return clf, report, w is not None

    def _fit_mlp(self, clf, tr: np.ndarray, w: np.ndarray | None, X: np.ndarray, y: np.ndarray):
        """Entraîne le mlp époque par époque et garde l'époque à la meilleure AP par chaîne sur
        MLP_HOLDOUT des chaînes de tr, jamais vues : la même question que val. Les deux classes
        pèsent autant au total, comme class_weight="balanced" de la logreg."""
        chans = np.array([self.rows[i]["source"] for i in tr])
        names = np.unique(chans)
        rng = np.random.default_rng(0)
        held = set(rng.choice(names, max(1, round(MLP_HOLDOUT * len(names))), replace=False)) if len(names) > 1 else set()
        out = np.isin(chans, list(held))
        if not (y[tr[out]].any() and (~y[tr[out]].astype(bool)).any()):
            out[:] = False  # chaînes mises de côté sans les deux classes : pas d'arrêt possible
        fit_i, w_fit = tr[~out], (np.ones(len(tr)) if w is None else w)[~out]
        yf = y[fit_i]
        for c in (0, 1):  # même calcul que class_weight="balanced" : sur les effectifs, puis x weights
            m = yf == c
            if m.any():
                w_fit = np.where(m, w_fit * len(yf) / (2 * m.sum()), w_fit)
        *prep, (_, mlp) = clf.steps
        Xf, Xo = X[fit_i], X[tr[out]]
        for _, step in prep:
            Xf = step.fit_transform(Xf)
            Xo = step.transform(Xo) if out.any() else Xo
        best, best_ap, stale = None, -1.0, 0
        for _ in range(MLP_MAX_EPOCHS):
            mlp.partial_fit(Xf, yf, sample_weight=w_fit, classes=np.array([0, 1]))
            if not out.any():
                continue
            p = mlp.predict_proba(Xo)[:, 1]
            ap = metrics.group_ap(y[tr[out]], p, chans[out])
            ap = average_precision_score(y[tr[out]], p) if ap is None else ap
            if ap > best_ap + 1e-4:
                best, best_ap, stale = pickle.dumps(mlp), ap, 0
            else:
                stale += 1
                if stale >= MLP_PATIENCE:
                    break
        if best is not None:
            clf.steps[-1] = (clf.steps[-1][0], pickle.loads(best))

    def _balance_groups(self, r: dict) -> set[str]:
        """Groupes de rééquilibrage d'une frame (voir BALANCE_GROUPS)."""
        groups = {self._orientation(r), "positif" if r["label_any"] else "negatif"}
        if r["sample_id"] in self._empty_samples():
            groups.add("clip_vide")
        return groups

    def _balance(self, tr: np.ndarray, balance: dict) -> tuple[np.ndarray, dict]:
        """Rééchantillonne les images de train : balance = {groupe: facteur}. Sous 1, on garde cette
        fraction des images ; au-dessus, chaque image est dupliquée (1.5 = une image sur deux en double,
        3 = trois exemplaires). Une image dans plusieurs groupes prend le plus petit facteur. Tirage
        déterministe (graine 0)."""
        if not balance:
            return tr, {}
        unknown = set(balance) - set(BALANCE_GROUPS)
        if unknown:
            raise ValueError(f"groupes inconnus {sorted(unknown)} ; groupes possibles : {BALANCE_GROUPS}")
        if any(not 0 <= float(v) <= MAX_FACTOR for v in balance.values()):
            raise ValueError(f"chaque facteur de balance doit être entre 0 et {MAX_FACTOR}")
        rng = np.random.default_rng(0)
        keep, before, after = [], Counter(), Counter()
        for i in tr:
            groups = self._balance_groups(self.rows[i]) & set(balance)
            factor = min((float(balance[g]) for g in groups), default=1.0)
            copies = int(factor) + (rng.random() < factor - int(factor))
            for g in groups:
                before[g] += 1
                after[g] += copies
            keep += [i] * copies
        if not keep:
            raise ValueError("le rééquilibrage ne laisse aucune image de train")
        report = {g: f"{after[g]} images pour {before[g]} au départ" for g in sorted(balance)}
        report["images_train"] = f"{len(keep)}/{len(tr)}"
        return np.array(keep, dtype=int), report

    def _weights(self, tr: np.ndarray, weights: dict) -> np.ndarray | None:
        """Poids des images de train dans la perte : weights = {groupe: poids}. Une image dans
        plusieurs groupes prend le plus grand poids, les autres gardent 1."""
        if not weights:
            return None
        unknown = set(weights) - set(BALANCE_GROUPS)
        if unknown:
            raise ValueError(f"groupes inconnus {sorted(unknown)} ; groupes possibles : {BALANCE_GROUPS}")
        if any(not 0 < float(v) <= MAX_FACTOR for v in weights.values()):
            raise ValueError(f"chaque poids doit être > 0 et au plus {MAX_FACTOR}")
        return np.array([max((float(weights[g]) for g in self._balance_groups(self.rows[i]) & set(weights)),
                             default=1.0) for i in tr])

    def _data_snapshot(self, split_id: str) -> dict:
        """Volume de données au moment de l'entraînement, pour comparer les expériences entre elles."""
        out = {"fps": FPS, "split_id": split_id}
        for split in SPLITS:
            idx = self._idx(split, split_id)
            out[split] = {"videos": len({self.rows[i]["sample_id"] for i in idx}),
                          "images_etiquetees": len(idx)}
        return out

    def _next_exp_id(self) -> str:
        # Plus grand numéro + 1, pas le nombre de dossiers : des expériences ont été supprimées.
        n = max((int(p.name[4:]) for p in EXP_DIR.glob("exp_*") if p.is_dir() and p.name[4:].isdigit()),
                default=0)
        return f"exp_{n + 1:03d}"

    def _test_chains(self, split_id: str) -> int:
        """Chaînes du test d'un split : la taille des jeux tirés pour le seuil prudent."""
        if split_id not in self._test_chains_cache:
            self._test_chains_cache[split_id] = len({self.rows[i]["source"] for i in self._idx("test", split_id)})
        return self._test_chains_cache[split_id]

    def _prudent_stale(self, op, split_id: str, has_groups: bool) -> bool:
        """Point de fonctionnement val calculé avec un autre réglage du seuil prudent que l'actuel
        (avant le 07/11/2026 : jeux de la taille du val, centile 80). Sans chaînes connues, le
        seuil n'est pas prudent et rien n'est périmé."""
        if not has_groups:
            return False
        expected = metrics.prudent_method(self._test_chains(split_id))
        return not isinstance(op, dict) or op.get("seuil_prudent") != expected

    def _val_at(self, experiment_id: str, precision: float):
        """Prédictions val d'une expérience et son point de fonctionnement à la précision visée."""
        d = EXP_DIR / experiment_id
        preds = json.loads((d / "val_preds.json").read_text(encoding="utf-8"))
        y, probs, groups = np.array(preds["y"]), np.array(preds["probs"]), preds.get("groups")
        split_id = json.loads((d / "config.json").read_text(encoding="utf-8")).get("split_id", LEGACY_SPLIT)
        # Seuil déjà calculé à l'entraînement pour les précisions usuelles (le bootstrap du seuil
        # prudent coûte) ; sinon, ou calculé avec un ancien réglage, recalculé et réécrit dans
        # metrics.json, pour que list_experiments montre le même point que celui testé.
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        key = f"rappel_a_precision_{precision}"
        known = m["val"].get(key)
        if isinstance(known, dict) and not self._prudent_stale(known, split_id, bool(groups)):
            op = metrics.operating_point(y, probs, known["threshold"])
        else:
            n = self._test_chains(split_id) if groups else None
            op = metrics.recall_at_precision(y, probs, precision, groups, n)
            if key in m["val"] and groups:
                m["val"][key] = op or "précision non atteinte"
                write_atomic(d / "metrics.json", json.dumps(_jsonable(m), indent=2))
        if op is None:
            raise ValueError(f"précision {precision} jamais atteinte sur val par {experiment_id}")
        return preds["frames"], y, probs, op

    def _orientation(self, r: dict) -> str:
        v = self._ann(r["sample_id"])["video"]
        return "portrait" if v["height"] > v["width"] else "paysage"
