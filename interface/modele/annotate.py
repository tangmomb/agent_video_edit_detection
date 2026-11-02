"""Annotation d'images de vraies vidéos envoyées dans l'interface, par les trois LLM annotateurs
du dataset (video_source.annotators : une image n'a d'étiquette que s'ils sont tous d'accord), pour
mesurer le modèle sur une vidéo choisie. Ces étiquettes servent à l'évaluation seulement : le
dataset d'entraînement est collecté par model_training_pipeline (generate_data).

Une vidéo annotée est rangée dans data/real_videos/<id>/ : video.*, frames/, labels.json (étiquette
des annotateurs et correction manuelle éventuelle de chaque image) et preds.json (notes du modèle).
Le banc interface/llm_bench compare les modèles candidats sur cette même annotation.
"""

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from video_source import annotators
from video_source.config import DATA_DIR

REAL_DIR = DATA_DIR / "real_videos"
MODEL = "accord de " + ", ".join(annotators.ANNOTATORS)
MAX_USD = 0.50  # plafond par vidéo : l'annotation s'arrête avant de le dépasser
WORKERS = 8  # images annotées en parallèle


def annotate(images: list[dict], apis: dict | None = None, max_usd: float | None = MAX_USD) -> dict:
    """images : [{"id": int, "path": Path}] ; renvoie les étiquettes dans le même ordre (ajout :
    True / False, None en cas de désaccord des annotateurs), le coût et la durée."""
    apis = apis if apis is not None else {}
    labels, spent = [], 0.0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(WORKERS) as pool:
        for i in range(0, len(images), WORKERS):
            chunk = images[i:i + WORKERS]
            # le prochain lot coûterait à peu près autant par image que les précédents
            if max_usd is not None and labels and spent + spent / len(labels) * len(chunk) > max_usd:
                break
            for im, v in zip(chunk, pool.map(lambda im: annotators.vote(im["path"], apis), chunk)):
                spent += v["cost_usd"]
                labels.append({"id": im["id"], "ajout": None if v["label"] is None else bool(v["label"]),
                               "types": [], "manquant": False,
                               "description": " · ".join(f"{m} : {'ajout' if a else 'rien' if a is False else '?'}"
                                                         for m, a in v["votes"].items())})
    return {"labels": labels, "model": MODEL, "cost_usd": round(spent, 4),
            "duration_s": round(time.perf_counter() - t0, 1), "complete": len(labels) == len(images)}


def annotate_video(result: dict, job: Path, apis: dict | None = None) -> dict:
    """Étiquettes des images d'une analyse de l'interface (result.json d'un job)."""
    frames = result["frames"]
    images = [{"id": i, "path": job / "frames" / f["url"].rsplit("/", 1)[1]} for i, f in enumerate(frames)]
    out = annotate(images, apis)
    for lab in out["labels"]:
        lab["t"] = frames[lab.pop("id")]["t"]
    return out


def save(job: Path, result: dict, annotation: dict) -> Path:
    """Range la vidéo, ses images, les notes du modèle et les étiquettes dans data/real_videos/."""
    dest = REAL_DIR / job.name
    (dest / "frames").mkdir(parents=True, exist_ok=True)
    for f in (job / "frames").glob("*.jpg"):
        shutil.copy2(f, dest / "frames" / f.name)
    for v in job.glob("video.*"):
        shutil.copy2(v, dest / v.name)
    (dest / "preds.json").write_text(json.dumps(result), encoding="utf-8")
    labels = {"model": annotation["model"], "cost_usd": annotation["cost_usd"] or 0.0,
              "complete": annotation["complete"],
              "frames": [{**lab, "manuel": None} for lab in annotation["labels"]]}
    (dest / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    return dest


def set_manual(job_id: str, i: int, value: bool | None) -> dict:
    """Correction manuelle de l'image i (None = revenir à l'étiquette des annotateurs)."""
    path = REAL_DIR / Path(job_id).name / "labels.json"
    labels = json.loads(path.read_text(encoding="utf-8"))
    labels["frames"][i]["manuel"] = value
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    return labels


def load(job_id: str) -> dict | None:
    path = REAL_DIR / Path(job_id).name / "labels.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
