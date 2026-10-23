"""Laboratoire du classifieur d'images « la frame contient un ajout de montage » : toutes les
opérations sur les données, les modèles et les évaluations, sans aucune dépendance aux agents.
Les agents (package agents/) n'exposent qu'un sous-ensemble de ces méthodes comme outils.

Règles :
- cible binaire label_any (au moins un ajout visible sur l'image) ;
- métrique principale : rappel à précision fixée (95 % par défaut), seuil choisi sur val ;
- la génération ignore les splits ; un split est un découpage train/val/test nommé et figé,
  fait par rush brut (make_split, stockés dans data/splits/) : les vidéos générées après sa
  création n'y entrent pas. Chaque expérience retient son split : on ne compare que des
  expériences faites sur le même split. L'ancien découpage « hash » (hash du rush) n'est plus
  utilisable : il grossissait à chaque génération, donc son val changeait sans changer de nom ;
- les décisions se prennent sur *val* ; *test* n'est accessible que par final_test, en
  nombre limité par session ;
- chaque entraînement est enregistré dans model_results/<exp_id>/ (à la racine du projet).
"""

import base64
import io
import json
import pickle
import time
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from edit_generator import annotations
from edit_generator.config import DATA_DIR, ROOT, SAMPLES_DIR
from edit_generator.generator import ELEMENT_TYPES
from edit_generator.ratelimit import QuotaExceeded, api_quotas

from . import metrics
from .features import extraction_cost, get_features
from .frames import FPS, FRAMES_DIR, build, load_index, source_key

EXP_DIR = ROOT / "model_results"
DATA_REQUESTS = DATA_DIR / "data_requests.jsonl"  # historique des générations, à côté des vidéos
SPLITS_DIR = DATA_DIR / "splits"
# Clips du test, figés au premier split : les splits suivants ne refont que train/val, et les
# vidéos générées ensuite n'entrent jamais en test (le Model Researcher ne peut pas l'influencer).
TEST_SET = SPLITS_DIR / "test_set.json"
# Clips retirés des splits créés ensuite (défaut de rendu, doublon) : exclude_clips, Model Researcher.
EXCLUDED = DATA_DIR / "excluded.json"
# Découpage d'avant make_split, non figé : refusé pour entraîner et tester.
LEGACY_SPLIT = "hash"
SPLITS = ("train", "val", "test")
# Caractéristiques des vidéos sur lesquelles make_split peut équilibrer les splits.
STRATIFY_KEYS = ("type", "orientation", "clip_vide")
MODELS = ("logreg", "mlp", "hgb")
# Groupes d'images qu'on peut sous-échantillonner dans train (train_model, paramètre balance) :
# un type d'élément = images positives dont tous les ajouts visibles sont de ce type ;
# negatif = images sans ajout ; clip_vide = images des clips sans aucun ajout ; orientation.
BALANCE_GROUPS = (*ELEMENT_TYPES, "negatif", "clip_vide", "portrait", "paysage")
MAX_QUOTA_FAILURES = 3  # échecs de quota d'affilée avant d'interrompre une génération


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


def _compact_val(val: dict) -> dict:
    """Val au point de précision par défaut (95 %) : seuil, rappel, erreurs. La précision n'est
    pas reprise (au moins 95 % par construction), les volumes du val sont ceux du split."""
    out = {"auc": val["auc"], "ap": val["ap"]}
    op = val[f"rappel_a_precision_{metrics.DEFAULT_PRECISION}"]
    if not isinstance(op, dict):
        return {**out, "p95": op}  # "précision non atteinte"
    return {**out, "seuil_p95": op["threshold"], "recall": op["recall"], "fp": op["fp"], "fn": op["fn"]}


def _compact(e: dict, errors: dict | None = None) -> dict:
    """Expérience en format court pour les agents, sans son split (elle est rangée dessous) :
    config, hypothèse, val à 95 % et ses erreurs (errors, voir Lab._val_errors), test final. Le détail (90/99 %, volumes, durées) reste dans model_results/."""
    out = {"id": e["id"]} if "id" in e else {}
    out["embeddings"] = e["features"]
    out.update({k: e[k] for k in ("model", "temporal", "params") if k in e})
    if e.get("balance"):
        out["balance"] = e["balance"]
    out["hypothese"] = e.get("note", "")
    out["val"] = _compact_val(e["val"])
    if errors is not None:
        out["erreurs_val"] = errors
    if "final_test" in e:
        t, ft = e["final_test"]["test"], e["final_test"]
        out["test"] = {"recall": t["recall"], "precision": t["precision"], "fp": t["fp"], "fn": t["fn"],
                       "auc": ft["test_auc"], "ap": ft["test_ap"]}
    if e.get("en_production"):
        out["en_production"] = True
    return out


def _area_bucket(a: float) -> str:
    return "<1%" if a < 0.01 else "1-3%" if a < 0.03 else "3-8%" if a < 0.08 else ">8%"


def _has_box(el: dict) -> bool:
    """L'élément est posé sur un fond (encadré de texte, bandeau, carton plein écran)."""
    return bool(el["style"].get("box") or el["type"] == "lower_third")


def _orient(ann: dict) -> str:
    return "portrait" if ann["video"]["height"] > ann["video"]["width"] else "paysage"


def _video_traits(ann: dict, keys: list[str]) -> set[str]:
    """Caractéristiques d'une vidéo selon les clés de stratification (STRATIFY_KEYS)."""
    out = set()
    for k in keys:
        if k == "type":
            out |= {f"type:{el['type']}" for el in ann["elements"]}
        elif k == "orientation":
            out.add(f"orientation:{_orient(ann)}")
        elif k == "clip_vide" and not ann["elements"]:
            out.add("clip_vide")
    return out


class Lab:
    def __init__(self, data_budget: int = 250, final_test_budget: int = 2, max_per_request: int = 100):
        self.data_budget = data_budget
        self.max_per_request = max_per_request
        self.generated = 0
        self.final_tests_left = final_test_budget
        self._rows: list[dict] | None = None
        self._ann_cache: dict[str, dict] = {}
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
        """sample_id -> train/val/test. Les clips exclus à la création du split, ou générés après,
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
        return {r["sample_id"]: sp["clips"][r["source"]] for r in self.rows
                if r["source"] in sp["clips"] and r["sample_id"] not in excluded}

    def _excluded(self) -> dict[str, str]:
        return json.loads(EXCLUDED.read_text(encoding="utf-8")) if EXCLUDED.exists() else {}

    def _test_clips(self) -> set[str] | None:
        """Clips du test figé ; à défaut, ceux du premier split existant (splits d'avant test_set)."""
        if TEST_SET.exists():
            return set(json.loads(TEST_SET.read_text(encoding="utf-8"))["clips"])
        first = sorted(SPLITS_DIR.glob("split_*.json"))
        if not first:
            return None
        sp = json.loads(first[0].read_text(encoding="utf-8"))
        clips = sorted(c for c, v in sp["clips"].items() if v == "test")
        TEST_SET.write_text(json.dumps({"from": sp["split_id"], "clips": clips}, indent=2), encoding="utf-8")
        return set(clips)

    def _idx(self, split: str, split_id: str) -> np.ndarray:
        """Images du split (celles sur lesquelles on entraîne et on mesure)."""
        m = self._split_map(split_id)
        return np.array([i for i, r in enumerate(self.rows)
                         if m.get(r["sample_id"]) == split], dtype=int)

    def _y(self) -> np.ndarray:
        return np.array([r["label_any"] for r in self.rows])

    def _prev(self) -> np.ndarray:
        prev = np.full(len(self.rows), -1)
        last: dict[str, int] = {}
        for i in sorted(range(len(self.rows)), key=lambda i: (self.rows[i]["sample_id"], self.rows[i]["t"])):
            prev[i] = last.get(self.rows[i]["sample_id"], -1)
            last[self.rows[i]["sample_id"]] = i
        return prev

    def _ann(self, sample_id: str) -> dict:
        if sample_id not in self._ann_cache:
            self._ann_cache[sample_id] = annotations.load(SAMPLES_DIR / sample_id)
        return self._ann_cache[sample_id]

    def _X(self, features: list[str], temporal: str, prev: np.ndarray) -> tuple[np.ndarray, float]:
        mats, ms = [], 0.0
        for spec in features:
            ext, _, view = spec.partition(":")
            mats.append(get_features(self.rows, ext, view or "full"))
            ms += extraction_cost(ext, view or "full") or 0.0
        X = np.concatenate(mats, axis=1)
        if temporal == "diff":
            X_prev = np.where(prev[:, None] >= 0, X[np.maximum(prev, 0)], 0.0)
            X = np.concatenate([X, X - X_prev], axis=1)
        elif temporal != "none":
            raise ValueError("temporal doit être 'none' ou 'diff'")
        return X, ms

    # ------------------------------------------------------------------ outils
    def _composition(self, sids: set[str], with_positives: bool = True) -> dict:
        """Composition d'un ensemble de vidéos : volumes et répartitions des éléments."""
        rows = [r for r in self.rows if r["sample_id"] in sids]
        anns = [self._ann(s) for s in sorted(sids)]
        els = [(a, el) for a in anns for el in a["elements"]]

        def count(key):
            return dict(Counter(key(a, el) for a, el in els).most_common())

        out = {"rushes_bruts_uniques": len({r["source"] for r in rows}), "clips_generes": len(sids),
               "images_extraites_des_clips": len(rows),
               "clips_sans_ajout": sum(not a["elements"] for a in anns),
               "clips_avec_ajout": sum(bool(a["elements"]) for a in anns),
               "clips_par_orientation": dict(Counter(_orient(a) for a in anns)),
               "elements_par_type": count(lambda a, el: el["type"]),
               "elements_par_surface": count(lambda a, el: _area_bucket(
                   el["bbox"][2] * el["bbox"][3] / (a["video"]["width"] * a["video"]["height"]))),
               "elements_par_fond": count(lambda a, el: "avec fond" if _has_box(el) else "sans fond")}
        if with_positives:
            out["pourcentage_images_avec_ajout"] = (
                round(100 * sum(r["label_any"] for r in rows) / len(rows), 1) if rows else None)
        return out

    def _split_summaries(self) -> list[dict]:
        out = []
        all_sids = {r["sample_id"] for r in self.rows}
        for sid in sorted(p.stem for p in SPLITS_DIR.glob("split_*.json")):
            m = self._split_map(sid)
            sp = json.loads((SPLITS_DIR / f"{sid}.json").read_text(encoding="utf-8"))
            out.append({"split_id": sid, **{s: sum(v == s for v in m.values()) for s in SPLITS},
                        "note": sp["note"], "date": sp["date"], "params": sp["params"],
                        "clips_exclus": len(sp["excluded"]),
                        # vidéos générées après la création du split
                        "clips_hors_split": len(all_sids - set(m) - set(sp["excluded"]))})
        return out

    def dataset_info(self, split_id: str | None = None) -> dict:
        rows = self.rows
        budgets = {"clips_generables_restants": self.data_budget - self.generated,
                   "final_tests_restants": self.final_tests_left, "quotas_api": api_quotas()}
        if not rows:
            return {"dataset_vide": True, "budgets": budgets,
                    "message": "Aucune vidéo annotée : le dataset est à constituer (generate_data)."}
        sids = sorted({r["sample_id"] for r in rows})
        info = {"fps": FPS, "dataset": self._composition(set(sids)), "budgets": budgets}
        if split_id:
            m = self._split_map(split_id)
            info["composition_du_split"] = {
                "split_id": split_id,
                **{s: self._composition({k for k, v in m.items() if v == s}, with_positives=s != "test")
                   for s in SPLITS}}
        clips = Counter(source_key(self._ann(s)) for s in sids)
        info["diversite"] = {
            "rushes_bruts_uniques": len(clips), "clips_generes": len(sids),
            # Vidéos générées avant que generate_data n'écarte les rushes déjà utilisés.
            "rushes_bruts_reutilises": {k: n for k, n in clips.most_common() if n > 1},
        }
        return _jsonable(info)

    def _used_sources(self) -> set[str]:
        return {source_key(annotations.load(p.parent)) for p in SAMPLES_DIR.glob("s*/annotations.json")}

    def generate_data(self, n: int, reason: str, query: str | None = None, orientation: str | None = None,
                      type_weights: dict | None = None, lang: str = "mix",
                      empty_prob: float | None = None) -> dict:
        from edit_generator.generator import GenerationConfig, generate_sample
        from edit_generator.sources import default_photo_source, default_video_source

        n = min(int(n), self.max_per_request, self.data_budget - self.generated)
        if n <= 0:
            return {"error": "budget de génération épuisé pour cette session"}
        # Un rush brut ne sert qu'une fois : deux vidéos sur le même fond gonflent le dataset sans
        # le diversifier, et le split se fait par rush.
        used = self._used_sources()

        def accept(src: dict) -> bool:
            return source_key({"source": src}) not in used

        seeds = [int(p.name[1:]) for p in SAMPLES_DIR.glob("s*") if p.name[1:].isdigit()]
        start = max(seeds, default=0) + 1
        cfg = GenerationConfig(lang=lang)
        if empty_prob is not None:
            cfg.empty_prob = float(empty_prob)
        if type_weights:
            cfg.type_weights = {**cfg.type_weights, **{k: float(v) for k, v in type_weights.items()}}
        videos, photos = default_video_source(), default_photo_source()
        ok, errors, quota_streak, stopped = 0, [], 0, None
        for seed in range(start, start + n):
            try:
                out = generate_sample(seed, SAMPLES_DIR, cfg, query=query, orientation=orientation,
                                      video_source=videos, photo_source=photos, accept_source=accept)
                used.add(source_key(annotations.load(out)))
                ok += 1
                quota_streak = 0
            except QuotaExceeded as e:
                errors.append(str(e)[:120])
                quota_streak += 1
                if quota_streak >= MAX_QUOTA_FAILURES:
                    stopped = f"arrêt après {quota_streak} échecs de quota d'affilée : {e}"
                    break
            except Exception as e:
                errors.append(str(e)[:120])
        self.generated += ok  # seules les vidéos réussies consomment le budget
        build()
        self._rows = None
        with open(DATA_REQUESTS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"n": n, "ok": ok, "query": query, "orientation": orientation,
                                 "type_weights": type_weights, "lang": lang, "empty_prob": cfg.empty_prob, "reason": reason, "time": time.time()}) + "\n")
        return {"generees": ok, "echecs": len(errors), "exemples_erreurs": errors[:3],
                "arret": stopped,
                "images_total": len(self.rows), "budget_restant": self.data_budget - self.generated,
                "quotas_api": api_quotas()}

    def data_requests(self) -> list[dict]:
        path = DATA_REQUESTS
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def data_requested_history(self) -> dict:
        """Historique des générations regroupé par requête Pixabay : une requête où on obtient moins
        de clips que demandé est en voie d'épuisement (ses rushes ne servent qu'une fois)."""
        out: dict[str, dict] = {}
        for r in self.data_requests():
            label = r.get("query") or ("rushes en cache" if r.get("source") == "cache" else "aléatoire")
            t = out.setdefault(label,
                               {"appels": 0, "clips_demandes": 0, "clips_generes": 0, "orientations": []})
            t["appels"] += 1
            t["clips_demandes"] += r["n"]
            t["clips_generes"] += r["ok"]
            o = r.get("orientation") or "aléatoire"
            if o not in t["orientations"]:
                t["orientations"].append(o)
        return out

    def exclude_clips(self, clips: dict) -> dict:
        """Retire des clips des splits créés ensuite : {sample_id: raison}. Les splits existants ne
        changent pas, pour qu'une exclusion ne puisse pas rendre un val déjà mesuré plus facile.
        Réservé aux défauts constatés sur le clip lui-même (rendu cassé, doublon), jamais aux
        erreurs d'un modèle. Les fichiers ne sont pas supprimés."""
        all_sids = {r["sample_id"] for r in self.rows}
        if not clips:
            raise ValueError("aucun clip à exclure")
        if set(clips) - all_sids:
            raise ValueError(f"clips inconnus : {sorted(set(clips) - all_sids)}")
        if any(not str(v).strip() for v in clips.values()):
            raise ValueError("chaque exclusion doit avoir une raison")
        excluded = {**self._excluded(), **{k: str(v) for k, v in clips.items()}}
        EXCLUDED.write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"clips_exclus": excluded,
                "effet": "appliqué aux splits créés à partir de maintenant (make_split) ; splits existants inchangés"}

    def make_split(self, note: str, val_ratio: float = 0.2, test_ratio: float = 0.2,
                   stratify: list[str] | None = None, seed: int = 0) -> dict:
        """Crée un split train/val/test figé. Toutes les vidéos d'un même rush brut vont du même
        côté. Chaque caractéristique de stratify (type d'élément, orientation, clip vide) est
        répartie au plus près des proportions visées (stratification itérative, caractéristiques
        les plus rares d'abord) ; par défaut toutes, [] pour un tirage purement aléatoire.
        Le test est figé au premier split (TEST_SET) : ensuite, test_ratio est ignoré et seuls
        train et val sont redécoupés."""
        # Par défaut tout est stratifié : sans ça, un cas rare peut tomber entièrement en test et
        # jamais en val (run 004).
        stratify = list(STRATIFY_KEYS) if stratify is None else list(stratify)
        exclude = self._excluded()
        if not note.strip():
            raise ValueError("note obligatoire : pourquoi ce split")
        if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio <= 0.6):
            raise ValueError("val_ratio et test_ratio doivent être > 0, avec val + test <= 0.6")
        if set(stratify) - set(STRATIFY_KEYS):
            raise ValueError(f"stratify parmi {STRATIFY_KEYS}")
        all_sids = {r["sample_id"] for r in self.rows}
        frozen = self._test_clips()

        clips: dict[str, set[str]] = {}  # rush brut -> caractéristiques de ses vidéos
        for sid in sorted(all_sids - set(exclude)):
            ann = self._ann(sid)
            clips.setdefault(source_key(ann), set()).update(_video_traits(ann, stratify))
        if len(clips) < 3:
            raise ValueError("il faut au moins 3 rushes bruts pour faire un split")
        ratios = {"train": 1 - val_ratio - test_ratio, "val": val_ratio, "test": test_ratio}
        total = Counter(t for traits in clips.values() for t in traits)
        rng = np.random.default_rng(seed)
        order = list(clips)
        rng.shuffle(order)
        order.sort(key=lambda c: min((total[t] for t in clips[c]), default=10 ** 9))
        n_clips, got, assign = len(clips), Counter(), {}
        cap = {s: max(1, round(ratios[s] * n_clips)) for s in SPLITS}  # volumes visés, en clips
        got_t: dict[str, Counter] = {s: Counter() for s in SPLITS}
        if frozen is not None:  # test imposé : ses clips d'abord, le reste en train/val
            for c in [c for c in order if c in frozen]:
                assign[c] = "test"
                got["test"] += 1
                got_t["test"].update(clips[c])
            order = [c for c in order if c not in frozen]
            cap["test"] = got["test"]
        cap["train"] = n_clips - cap["val"] - cap["test"]
        for c in order:
            def deficit(s):
                # Déficits relatifs (rapportés à la cible) : en absolu, un cas rare (vu une seule
                # fois) partirait toujours en train, le split le plus gros.
                trait = sum(1 - got_t[s][t] / (ratios[s] * total[t]) for t in clips[c])
                return trait, 1 - got[s] / (ratios[s] * n_clips), rng.random()
            s = max([s for s in SPLITS if got[s] < cap[s] and (s != "test" or frozen is None)], key=deficit)
            assign[c] = s
            got[s] += 1
            got_t[s].update(clips[c])

        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        split_id = f"split_{len(list(SPLITS_DIR.glob('split_*.json'))) + 1:03d}"
        params = {"val_ratio": val_ratio, "test_ratio": None if frozen is not None else test_ratio,
                  "test_fige": frozen is not None, "stratify": stratify, "seed": seed}
        (SPLITS_DIR / f"{split_id}.json").write_text(json.dumps(
            {"split_id": split_id, "note": note, "date": time.strftime("%Y-%m-%d %H:%M"), "params": params,
             "excluded": exclude, "clips": assign}, ensure_ascii=False, indent=2), encoding="utf-8")
        if frozen is None:
            self._test_clips()  # fige le test de ce premier split
        m = self._split_map(split_id)
        return _jsonable({"split_id": split_id, "params": params, "clips_exclus": len(exclude),
                          **{s: self._composition({k for k, v in m.items() if v == s},
                                                  with_positives=s != "test") for s in SPLITS}})

    def production(self) -> dict | None:
        path = EXP_DIR / "production.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def train_model(self, embeddings: list[str], model: str = "logreg", temporal: str = "none",
                    params: dict | None = None, balance: dict | None = None, note: str = "",
                    split_id: str | None = None) -> dict:
        if not split_id:
            raise ValueError("split_id obligatoire : un split créé par make_split (voir list_experiments)")
        if model not in MODELS:
            raise ValueError(f"model dans {MODELS}")
        tr, va = self._idx("train", split_id), self._idx("val", split_id)
        if not len(tr) or not len(va):
            raise ValueError("splits train/val vides : il faut d'abord générer des données")
        params = params or {}
        tr, balance_report = self._balance(tr, balance or {})
        X, feat_ms = self._X(embeddings, temporal, self._prev())
        y = self._y()

        clf = self._make(model, params)
        t0 = time.time()
        clf.fit(X[tr], y[tr])
        train_s = time.time() - t0
        t0 = time.time()
        probs = clf.predict_proba(X[va])[:, 1]
        pred_ms = 1000 * (time.time() - t0) / len(va)

        exp_id = self._next_exp_id()
        result = {
            "val": metrics.summary(y[va], probs), "train_s": train_s,
            "ms_par_frame": feat_ms + pred_ms, "n_features": X.shape[1],
            "n_train": len(tr), "n_val": len(va), "data": self._data_snapshot(split_id),
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        if balance_report:
            result["reequilibrage_train"] = balance_report
        # Clé "features" gardée dans config.json : l'interface et final_test la relisent.
        config = {"split_id": split_id, "features": embeddings, "model": model, "temporal": temporal,
                  "params": params, "balance": balance or {}, "note": note}
        d = EXP_DIR / exp_id
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (d / "metrics.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        # Prédictions val indexées par frame : restent valides si le dataset grossit ensuite.
        (d / "val_preds.json").write_text(json.dumps(
            {"frames": [self.rows[i]["frame"] for i in va], "y": y[va].tolist(),
             "probs": [round(float(p), 5) for p in probs]}), encoding="utf-8")
        with open(d / "model.pkl", "wb") as fh:
            pickle.dump(clf, fh)
        # Format court : la config est celle de l'appel, le détail reste dans metrics.json.
        short = {"val": _compact_val(result["val"])}
        if balance_report:
            short["reequilibrage_train"] = balance_report
        return _jsonable({"experiment_id": exp_id, **short})

    def analyze_errors(self, experiment_id: str, precision: float = metrics.DEFAULT_PRECISION) -> dict:
        frames, y, probs, op = self._val_at(experiment_id, precision)
        pred = probs >= op["threshold"]
        by_frame = {r["frame"]: r for r in self.rows}
        fn_rows = [by_frame[f] for f, yi, pi in zip(frames, y, pred) if yi and not pi]
        tp_rows = [by_frame[f] for f, yi, pi in zip(frames, y, pred) if yi and pi]
        fp_rows = [by_frame[f] for f, yi, pi in zip(frames, y, pred) if not yi and pi]

        def rates(key):
            tot, miss = Counter(), Counter()
            for r in tp_rows + fn_rows:
                for k in key(r):
                    tot[k] += 1
            for r in fn_rows:
                for k in key(r):
                    miss[k] += 1
            return {k: f"{miss[k]}/{tot[k]} ratées" for k in sorted(tot, key=str)}

        return _jsonable({
            "experiment_id": experiment_id, "precision_visee": precision, **op,
            "positifs_rates_par_type": rates(lambda r: {e["type"] for e in self._visible(r)}),
            "positifs_rates_par_plus_grande_surface": rates(
                lambda r: {_area_bucket(max(e["area_ratio"] for e in self._visible(r)))}),
            "positifs_rates_avec_fond": rates(
                lambda r: {"avec fond" if any(e["has_box"] for e in self._visible(r)) else "sans fond"}),
            "positifs_rates_par_nb_elements": rates(lambda r: {min(len(r["visible"]), 4)}),
            "positifs_rates_par_orientation": rates(lambda r: {self._orientation(r)}),
            "faux_positifs_par_orientation": dict(Counter(self._orientation(r) for r in fp_rows)),
            "faux_positifs_par_clip": dict(Counter(r["sample_id"] for r in fp_rows).most_common(8)),
            "exemples_rates": [{"image": r["frame"], "t": r["t"],
                                "elements": self._visible(r)} for r in fn_rows[:10]],
        })

    def show_examples(self, experiment_id: str, kind: str = "missed", n: int = 4,
                      precision: float = metrics.DEFAULT_PRECISION) -> list[dict]:
        """Renvoie des blocs image (frames val, ajouts encadrés en rouge) pour inspection visuelle."""
        frames, y, probs, op = self._val_at(experiment_id, precision)
        pred = probs >= op["threshold"]
        want = (y == 1) & ~pred if kind == "missed" else (y == 0) & pred
        by_frame = {r["frame"]: r for r in self.rows}
        label = "ajout raté" if kind == "missed" else "faux positif"
        blocks: list[dict] = []
        for i in np.flatnonzero(want)[:n]:
            r = by_frame[frames[i]]
            img = Image.open(FRAMES_DIR / r["frame"]).convert("RGB")
            ann = self._ann(r["sample_id"])
            sx, sy = img.width / ann["video"]["width"], img.height / ann["video"]["height"]
            draw = ImageDraw.Draw(img)
            for el in ann["elements"]:
                b = annotations.bbox_at(el, r["t"])
                if b and el["id"] in r["visible"]:
                    x, yy, w, h = b
                    draw.rectangle([x * sx, yy * sy, (x + w) * sx, (yy + h) * sy], outline="red", width=3)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80)
            blocks.append({"type": "text", "text": f"{label} {r['sample_id']} t={r['t']} p={probs[i]:.2f} "
                                                   f"(seuil {op['threshold']:.2f})"})
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                       "data": base64.b64encode(buf.getvalue()).decode()}})
        return blocks or [{"type": "text", "text": "aucun exemple"}]

    def list_experiments(self) -> list[dict]:
        """Tout ce qu'on sait de chaque expérience : config, métriques val, volume de données,
        test final et statut de production."""
        prod = self.production() or {}
        out = []
        for d in sorted(p for p in EXP_DIR.glob("exp_*") if p.is_dir()):
            if not (d / "metrics.json").exists():
                continue
            e = {"id": d.name,
                 **json.loads((d / "config.json").read_text(encoding="utf-8")),
                 **json.loads((d / "metrics.json").read_text(encoding="utf-8"))}
            if (d / "final_test.json").exists():
                e["final_test"] = json.loads((d / "final_test.json").read_text(encoding="utf-8"))
            e["en_production"] = prod.get("experiment_id") == d.name
            out.append(e)
        return out

    def experiments_overview(self, split_id: str | None = None) -> dict:
        """Ce que renvoie l'outil list_experiments : chaque split avec ses expériences en format
        court. Un split sans expérience n'apparaît que s'il est le plus récent (celui sur lequel
        entraîner)."""
        exps = self.list_experiments()
        summaries = self._split_summaries()
        latest = summaries[-1]["split_id"] if summaries else None
        y = self._y()
        out = []
        for sp in summaries:
            sid = sp["split_id"]
            mine = [_compact(e, self._val_errors(e["id"])) for e in exps if e.get("split_id") == sid]
            if (split_id and sid != split_id) or (not split_id and not mine and sid != latest):
                continue
            va = self._idx("val", sid)
            # Rushes par partie, recomptés sur les clips réellement assignés : rushes_shared (rushes
            # présents dans plusieurs parties) doit rester à 0, sinon val/test fuient dans train.
            m = self._split_map(sid)
            parts: dict[str, set[str]] = {}
            for r in self.rows:
                if r["sample_id"] in m:
                    parts.setdefault(r["source"], set()).add(m[r["sample_id"]])
            entry = {"split_id": sid,
                     **{f"rushes_{s}": sum(ps == {s} for ps in parts.values()) for s in SPLITS},
                     "rushes_shared": sum(len(ps) > 1 for ps in parts.values()),
                     **{f"clips_{s}": sp[s] for s in SPLITS},
                     "val_images": len(va), "val_images_positives": int(y[va].sum()),
                     "clips_hors_split": sp["clips_hors_split"]}
            if sp["clips_exclus"]:
                entry["clips_exclus"] = sp["clips_exclus"]
            note = sp["note"]
            entry["note"] = note if len(note) <= 120 else note[:120].rstrip() + "…"
            entry["experiences"] = mine
            out.append(entry)
        return {"splits": out}

    def final_test(self, experiment_id: str, precision: float = metrics.DEFAULT_PRECISION) -> dict:
        if self.final_tests_left <= 0:
            return {"error": "plus de final_test disponible dans cette session"}
        _, _, _, op = self._val_at(experiment_id, precision)
        d = EXP_DIR / experiment_id
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        split_id = config.get("split_id", LEGACY_SPLIT)
        te = self._idx("test", split_id)  # avant de décompter : un split refusé ne coûte pas de test
        self.final_tests_left -= 1
        with open(d / "model.pkl", "rb") as fh:
            clf = pickle.load(fh)
        X, _ = self._X(config["features"], config["temporal"], self._prev())
        y = self._y()[te]
        probs = clf.predict_proba(X[te])[:, 1]
        s = metrics.summary(y, probs)
        result = {"experiment_id": experiment_id, "split_id": split_id,
                  "precision_visee": precision, "seuil_choisi_sur_val": op["threshold"], "val": op,
                  "test": metrics.operating_point(y, probs, op["threshold"]),
                  "test_auc": s["auc"], "test_ap": s["ap"], "final_tests_restants": self.final_tests_left}
        (d / "final_test.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        return _jsonable(result)

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
        (EXP_DIR / "production.json").write_text(json.dumps(_jsonable(prod), indent=2), encoding="utf-8")
        return _jsonable(prod)

    def select_production(self, precision: float = metrics.DEFAULT_PRECISION) -> dict:
        """Choix automatique du modèle de production, fait par le code en fin de run, parmi les
        expériences du plus gros split (le plus de clips ; à égalité, le plus récent) : celle de
        meilleur rappel sur val à la précision visée parmi celles qui ont réussi leur test final
        (précision visée tenue sur test). À défaut, la meilleure non testée, puis la meilleure
        tout court, marquées comme non vérifiées. Égalités départagées par AP, AUC puis temps de
        calcul. Écrit production.json (via save_model) et renvoie le classement."""
        splits = self._split_summaries()
        if not splits:
            return {"error": "aucun split figé : rien à mettre en production"}
        big = max(splits, key=lambda s: (sum(s[k] for k in SPLITS), s["split_id"]))["split_id"]
        ranked = []
        for e in self.list_experiments():
            if e.get("split_id") != big:
                continue
            _, y, probs, _ = self._val_at(e["id"], 0.0)  # toutes les prédictions val de l'expérience
            op = metrics.recall_at_precision(y, probs, precision)
            ranked.append({"experiment_id": e["id"], "val_recall": op["recall"] if op else 0.0,
                           "ap": e["val"]["ap"] or 0.0, "auc": e["val"]["auc"] or 0.0,
                           "ms_par_frame": e["ms_par_frame"], "test": self._test_status(e["id"], precision)})
        if not ranked:
            return {"error": f"aucune expérience sur le plus gros split ({big})"}
        ranked.sort(key=lambda r: (-r["val_recall"], -r["ap"], -r["auc"], r["ms_par_frame"]))
        order = {"réussi": 0, "non testé": 1, "échoué": 2}
        best = min(ranked, key=lambda r: order[r["test"]["statut"]])  # min garde l'ordre val à égalité
        verifie = best["test"]["statut"] == "réussi"
        prod = self.save_model(best["experiment_id"], precision,
                               f"choix automatique de fin de run sur {big} (le plus gros split) : "
                               + ("meilleur rappel val parmi les modèles qui ont réussi leur test final"
                                  if verifie else "NON VÉRIFIÉ sur test (aucun modèle n'a réussi son test "
                                  f"final), meilleur rappel val ({best['test']['statut']})"),
                               test=best["test"])
        return _jsonable({"split_id": big, "production": prod, "verifie_sur_test": verifie,
                          "classement": ranked})

    def _test_status(self, experiment_id: str, precision: float) -> dict:
        """Résultat du test final d'une expérience pour une précision visée : réussi si la
        précision visée est tenue sur test au seuil choisi sur val."""
        path = EXP_DIR / experiment_id / "final_test.json"
        if not path.exists():
            return {"statut": "non testé"}
        ft = json.loads(path.read_text(encoding="utf-8"))
        if ft["precision_visee"] != precision:
            return {"statut": "non testé", "detail": f"testé à une autre précision visée ({ft['precision_visee']})"}
        t = ft["test"]
        return {"statut": "réussi" if t["precision"] >= precision else "échoué",
                "recall": t["recall"], "precision": t["precision"]}

    # ------------------------------------------------------------------ interne
    @staticmethod
    def _make(model: str, params: dict):
        steps = [StandardScaler()]
        if params.get("pca"):
            steps.append(PCA(n_components=int(params["pca"]), random_state=0))
        if model == "logreg":
            clf = LogisticRegression(C=float(params.get("C", 0.1)), max_iter=3000, class_weight="balanced")
        elif model == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=tuple(params.get("hidden", [256])),
                                alpha=float(params.get("alpha", 1e-3)), early_stopping=True,
                                max_iter=300, random_state=0)
        else:
            clf = HistGradientBoostingClassifier(max_iter=int(params.get("max_iter", 200)),
                                                 learning_rate=float(params.get("lr", 0.1)),
                                                 max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
                                                 class_weight="balanced", random_state=0)
        return make_pipeline(*steps, clf)

    def _balance_groups(self, r: dict) -> set[str]:
        """Groupes de rééquilibrage d'une frame (voir BALANCE_GROUPS)."""
        ann = self._ann(r["sample_id"])
        groups = {self._orientation(r)}
        if not ann["elements"]:
            groups.add("clip_vide")
        if not r["label_any"]:
            groups.add("negatif")
        else:
            types = {el["type"] for el in ann["elements"] if el["id"] in r["visible"]}
            if len(types) == 1:
                groups.add(types.pop())
        return groups

    def _balance(self, tr: np.ndarray, balance: dict) -> tuple[np.ndarray, dict]:
        """Sous-échantillonne les frames de train : balance = {groupe: fraction gardée}. Une frame
        dans plusieurs groupes garde la plus petite fraction. Tirage déterministe (graine 0)."""
        if not balance:
            return tr, {}
        unknown = set(balance) - set(BALANCE_GROUPS)
        if unknown:
            raise ValueError(f"groupes inconnus {sorted(unknown)} ; groupes possibles : {BALANCE_GROUPS}")
        if any(not 0 <= float(v) <= 1 for v in balance.values()):
            raise ValueError("chaque fraction gardée doit être entre 0 et 1")
        rng = np.random.default_rng(0)
        keep, before, after = [], Counter(), Counter()
        for i in tr:
            groups = self._balance_groups(self.rows[i]) & set(balance)
            frac = min((float(balance[g]) for g in groups), default=1.0)
            kept = rng.random() < frac
            for g in groups:
                before[g] += 1
                after[g] += kept
            if kept:
                keep.append(i)
        if not keep:
            raise ValueError("le rééquilibrage ne laisse aucune image de train")
        report = {g: f"{after[g]}/{before[g]} images gardées" for g in sorted(balance)}
        report["images_train"] = f"{len(keep)}/{len(tr)}"
        return np.array(keep, dtype=int), report

    def _data_snapshot(self, split_id: str) -> dict:
        """Volume de données au moment de l'entraînement, pour comparer les expériences entre elles."""
        out = {"fps": FPS, "split_id": split_id}
        for split in SPLITS:
            idx = self._idx(split, split_id)
            out[split] = {"clips_generes": len({self.rows[i]["sample_id"] for i in idx}),
                          "images_extraites_des_clips": len(idx)}
        return out

    def _next_exp_id(self) -> str:
        # Plus grand numéro + 1, pas le nombre de dossiers : des expériences ont été supprimées.
        n = max((int(p.name[4:]) for p in EXP_DIR.glob("exp_*") if p.is_dir() and p.name[4:].isdigit()),
                default=0)
        return f"exp_{n + 1:03d}"

    def _val_at(self, experiment_id: str, precision: float):
        """Prédictions val d'une expérience et son point de fonctionnement à la précision visée."""
        preds = json.loads((EXP_DIR / experiment_id / "val_preds.json").read_text(encoding="utf-8"))
        y, probs = np.array(preds["y"]), np.array(preds["probs"])
        op = metrics.recall_at_precision(y, probs, precision)
        if op is None:
            raise ValueError(f"précision {precision} jamais atteinte sur val par {experiment_id}")
        return preds["frames"], y, probs, op

    def _val_errors(self, experiment_id: str) -> dict | None:
        """Résumé des erreurs val au seuil de précision par défaut, pour list_experiments : ajouts
        ratés par type (une image ratée compte pour chaque type visible), clips à fausses alertes.
        Le détail est dans analyze_errors."""
        try:
            frames, y, probs, op = self._val_at(experiment_id, metrics.DEFAULT_PRECISION)
        except ValueError:
            return None  # précision jamais atteinte
        by_frame = {r["frame"]: r for r in self.rows}
        pred = probs >= op["threshold"]
        fn, fp = Counter(), Counter()
        for f, yi, pi in zip(frames, y, pred):
            if f not in by_frame:
                continue
            if yi and not pi:
                fn.update({e["type"] for e in self._visible(by_frame[f])})
            elif pi and not yi:
                fp[by_frame[f]["sample_id"]] += 1
        return {"fn_par_type": dict(fn.most_common()),
                "fp_par_clip": dict(fp.most_common(5))}

    def _visible(self, r: dict) -> list[dict]:
        """Éléments visibles sur l'image, avec surface relative et présence d'un fond."""
        ann = self._ann(r["sample_id"])
        W, H = ann["video"]["width"], ann["video"]["height"]
        out = []
        for el in ann["elements"]:
            if el["id"] in r["visible"]:
                b = annotations.bbox_at(el, r["t"]) or el["bbox"]
                out.append({"type": el["type"], "text": el.get("text"),
                            "area_ratio": round(b[2] * b[3] / (W * H), 4),
                            "has_box": _has_box(el)})
        return out

    def _orientation(self, r: dict) -> str:
        v = self._ann(r["sample_id"])["video"]
        return "portrait" if v["height"] > v["width"] else "paysage"
