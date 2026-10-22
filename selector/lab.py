"""Laboratoire du classifieur d'images « la frame contient un ajout de montage » : toutes les
opérations sur les données, les modèles et les évaluations, sans aucune dépendance aux agents.
Les agents (package agents/) n'exposent qu'un sous-ensemble de ces méthodes comme outils.

Règles :
- cible binaire label_any (au moins un ajout pleinement visible) ; les frames ambiguës (un
  élément en fondu) sont exclues de l'entraînement et de l'évaluation ;
- métrique principale : rappel à précision fixée (95 % par défaut), seuil choisi sur val ;
- les décisions se prennent sur le split *val* ; le split *test* n'est accessible que par
  final_test, en nombre limité par session ;
- chaque entraînement est enregistré dans data/experiments/<exp_id>/.
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

from editgen import annotations
from editgen.config import DATA_DIR, SAMPLES_DIR
from editgen.ratelimit import QuotaExceeded, api_quotas

from . import metrics
from .features import EXTRACTORS, VIEWS, extraction_cost, get_features
from .frames import FPS, FRAMES_DIR, build, load_index, source_key, split_for

EXP_DIR = DATA_DIR / "experiments"
MODELS = ("logreg", "mlp", "hgb")
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


def _area_bucket(a: float) -> str:
    return "<1%" if a < 0.01 else "1-3%" if a < 0.03 else "3-8%" if a < 0.08 else ">8%"


class Lab:
    def __init__(self, data_budget: int = 300, final_test_budget: int = 2, max_per_request: int = 100):
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

    def _idx(self, split: str) -> np.ndarray:
        """Frames non ambiguës du split (celles sur lesquelles on entraîne et on mesure)."""
        return np.array([i for i, r in enumerate(self.rows) if r["split"] == split and not r["ambiguous"]],
                        dtype=int)

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
    def dataset_info(self) -> dict:
        rows = self.rows
        budgets = {"videos_generables_restantes": self.data_budget - self.generated,
                   "final_tests_restants": self.final_tests_left, "quotas_api": api_quotas()}
        if not rows:
            return {"dataset_vide": True, "budgets": budgets,
                    "message": "Aucune vidéo annotée : le dataset est à constituer (generate_data)."}
        y = self._y()
        info = {"fps": FPS, "splits": {}, "extractors": {}, "budgets": budgets}
        for split in ("train", "val", "test"):
            all_idx = [i for i, r in enumerate(rows) if r["split"] == split]
            if not all_idx:
                continue
            idx = self._idx(split)
            sids = {rows[i]["sample_id"] for i in all_idx}
            types = Counter(el["type"] for s in sids for el in self._ann(s)["elements"])
            entry = {"videos": len(sids), "sources": len({rows[i]["source"] for i in all_idx}),
                     "frames": len(idx), "frames_ambigues_exclues": len(all_idx) - len(idx),
                     "elements_par_type": dict(types)}
            if split != "test":
                entry["taux_positifs"] = float(y[idx].mean()) if len(idx) else None
            info["splits"][split] = entry
        for ext in EXTRACTORS:
            for view in VIEWS:
                ms = extraction_cost(ext, view)
                info["extractors"][f"{ext}:{view}"] = {"en_cache": ms is not None, "ms_par_frame": ms}
        return _jsonable(info)

    def generate_data(self, n: int, reason: str, query: str | None = None, orientation: str | None = None,
                      type_weights: dict | None = None, lang: str = "mix",
                      empty_prob: float | None = None, split: str | None = None) -> dict:
        from editgen.generator import GenerationConfig, generate_sample
        from editgen.sources import default_photo_source, default_video_source

        if split not in (None, "train", "val", "test"):
            return {"error": f"split inconnu : {split} (train, val ou test)"}
        n = min(int(n), self.max_per_request, self.data_budget - self.generated)
        if n <= 0:
            return {"error": "budget de génération épuisé pour cette session"}
        # Le split dépend de la vidéo source : pour cibler un split, on ne retient que les sources
        # qui y tombent.
        accept = (lambda src: split_for(source_key({"source": src})) == split) if split else None
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
                generate_sample(seed, SAMPLES_DIR, cfg, query=query, orientation=orientation,
                                video_source=videos, photo_source=photos, accept_source=accept)
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
        with open(EXP_DIR / "data_requests.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"n": n, "ok": ok, "query": query, "orientation": orientation, "split": split,
                                 "type_weights": type_weights, "reason": reason, "time": time.time()}) + "\n")
        return {"generees": ok, "echecs": len(errors), "exemples_erreurs": errors[:3],
                "arret": stopped,
                "frames_total": len(self.rows), "budget_restant": self.data_budget - self.generated,
                "quotas_api": api_quotas()}

    def data_requests(self) -> list[dict]:
        path = EXP_DIR / "data_requests.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def production(self) -> dict | None:
        path = EXP_DIR / "production.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def train_model(self, features: list[str], model: str = "logreg", temporal: str = "none",
                    params: dict | None = None, note: str = "") -> dict:
        if model not in MODELS:
            raise ValueError(f"model dans {MODELS}")
        tr, va = self._idx("train"), self._idx("val")
        if not len(tr) or not len(va):
            raise ValueError("splits train/val vides : il faut d'abord générer des données")
        params = params or {}
        X, feat_ms = self._X(features, temporal, self._prev())
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
            "n_train": len(tr), "n_val": len(va), "data": self._data_snapshot(),
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        config = {"features": features, "model": model, "temporal": temporal, "params": params, "note": note}
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
        return _jsonable({"experiment_id": exp_id, **result})

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
            "faux_positifs_par_video": dict(Counter(r["sample_id"] for r in fp_rows).most_common(8)),
            "exemples_rates": [{"frame": r["frame"], "t": r["t"],
                                "elements": [{k: e[k] for k in ("type", "text", "area_ratio", "has_box")}
                                             for e in self._visible(r)]} for r in fn_rows[:10]],
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

    def final_test(self, experiment_id: str, precision: float = metrics.DEFAULT_PRECISION) -> dict:
        if self.final_tests_left <= 0:
            return {"error": "plus de final_test disponible dans cette session"}
        _, _, _, op = self._val_at(experiment_id, precision)
        self.final_tests_left -= 1
        d = EXP_DIR / experiment_id
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        with open(d / "model.pkl", "rb") as fh:
            clf = pickle.load(fh)
        X, _ = self._X(config["features"], config["temporal"], self._prev())
        te = self._idx("test")
        y = self._y()[te]
        probs = clf.predict_proba(X[te])[:, 1]
        s = metrics.summary(y, probs)
        result = {"experiment_id": experiment_id, "precision_visee": precision,
                  "seuil_choisi_sur_val": op["threshold"], "val": op,
                  "test": metrics.operating_point(y, probs, op["threshold"]),
                  "test_auc": s["auc"], "test_ap": s["ap"], "final_tests_restants": self.final_tests_left}
        (d / "final_test.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        return _jsonable(result)

    def save_model(self, experiment_id: str, precision: float, note: str) -> dict:
        _, _, _, op = self._val_at(experiment_id, precision)
        prod = {"experiment_id": experiment_id, "precision_visee": precision, "threshold": op["threshold"],
                "val_recall": op["recall"], "val_precision": op["precision"], "note": note, "time": time.time()}
        (EXP_DIR / "production.json").write_text(json.dumps(_jsonable(prod), indent=2), encoding="utf-8")
        return _jsonable(prod)

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

    def _data_snapshot(self) -> dict:
        """Volume de données au moment de l'entraînement, pour comparer les expériences entre elles."""
        out = {"fps": FPS}
        for split in ("train", "val", "test"):
            idx = self._idx(split)
            out[split] = {"videos": len({self.rows[i]["sample_id"] for i in idx}), "frames": len(idx)}
        return out

    def _next_exp_id(self) -> str:
        n = len([p for p in EXP_DIR.glob("exp_*") if p.is_dir()])
        return f"exp_{n + 1:03d}"

    def _val_at(self, experiment_id: str, precision: float):
        """Prédictions val d'une expérience et son point de fonctionnement à la précision visée."""
        preds = json.loads((EXP_DIR / experiment_id / "val_preds.json").read_text(encoding="utf-8"))
        y, probs = np.array(preds["y"]), np.array(preds["probs"])
        op = metrics.recall_at_precision(y, probs, precision)
        if op is None:
            raise ValueError(f"précision {precision} jamais atteinte sur val par {experiment_id}")
        return preds["frames"], y, probs, op

    def _visible(self, r: dict) -> list[dict]:
        """Éléments pleinement visibles sur la frame, avec surface relative et présence d'un fond."""
        ann = self._ann(r["sample_id"])
        W, H = ann["video"]["width"], ann["video"]["height"]
        out = []
        for el in ann["elements"]:
            if el["id"] in r["visible"]:
                b = annotations.bbox_at(el, r["t"]) or el["bbox"]
                out.append({"type": el["type"], "text": el.get("text"),
                            "area_ratio": round(b[2] * b[3] / (W * H), 4),
                            "has_box": bool(el["style"].get("box") or el["type"] == "lower_third")})
        return out

    def _orientation(self, r: dict) -> str:
        v = self._ann(r["sample_id"])["video"]
        return "portrait" if v["height"] > v["width"] else "paysage"
