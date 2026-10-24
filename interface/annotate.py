"""Annotation d'images de vraies vidéos par un LLM de vision (Mistral), pour mesurer le modèle sur
du réel. Les étiquettes servent à l'évaluation seulement, jamais à l'entraînement.

Une vidéo annotée est rangée dans data/real_videos/<id>/ : video.*, frames/, labels.json (étiquette
du LLM et correction manuelle éventuelle de chaque image) et preds.json (notes du modèle).
Le banc utils/llm_bench compare plusieurs modèles Mistral sur cette même annotation.
"""

import base64
import io
import json
import os
import shutil
import time
from pathlib import Path

from PIL import Image

from agents.runner import LARGE, PRICES, _tokens
from edit_generator.config import DATA_DIR

REAL_DIR = DATA_DIR / "real_videos"
MODEL = LARGE
BATCH = 6  # images par requête
MAX_SIDE = 768  # plus grand côté envoyé ; le nombre de tokens d'une image suit son nombre de pixels
MAX_USD = 0.50  # plafond par vidéo : l'annotation s'arrête avant de le dépasser

PROMPT = """Tu annotes des images extraites de vidéos pour entraîner un détecteur d'ajouts de montage.

Un ajout de montage est un élément graphique posé PAR-DESSUS l'image en post-production : texte \
incrusté (mot-clé, titre, question, sous-titre), bandeau nom/fonction (lower third), image ou photo \
incrustée, forme ou pictogramme, carton plein écran, logo ou filigrane de chaîne ou de marque, \
compteur, flèche, émoji. Ce n'est PAS un ajout : ce qui fait partie de la scène filmée (enseigne, \
affiche, écran, texte sur un vêtement ou un objet, panneau), même si c'est du texte.

Pour chaque image, dans l'ordre, réponds uniquement par un objet JSON de la forme :
{"images": [{"id": <numéro de l'image>, "ajout": true|false, "types": ["mot-clé" | "titre" | \
"question" | "sous-titre" | "bandeau" | "image" | "forme" | "carton" | "logo" | "autre"], \
"description": "<court>"}]}
"types" est vide quand "ajout" vaut false. Dans le doute sur un élément très petit ou à moitié \
visible, décris-le et tranche."""


def _data_url(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _client():
    from mistralai.client import Mistral
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def cost(model: str, tokens: dict) -> float | None:
    """Coût en $ d'après les tarifs connus (agents.runner.PRICES) ; None si le modèle n'y est pas."""
    p = PRICES.get(model)
    if p is None:
        return None
    fresh = tokens["input"] - tokens["cached_input"]
    return (fresh * p["input"] + tokens["cached_input"] * p.get("cached_input", p["input"])
            + tokens["output"] * p["output"]) / 1e6


def annotate(images: list[dict], client=None, max_usd: float | None = MAX_USD, model: str = MODEL,
             max_side: int = MAX_SIDE, reasoning: str | None = None, batch: int = BATCH) -> dict:
    """images : [{"id": int, "path": Path}] ; renvoie les étiquettes dans le même ordre, avec les
    tokens consommés, le coût (si le tarif du modèle est connu) et la durée."""
    client = client or _client()
    labels, spent, last = [], 0.0, 0.0
    tokens = {"input": 0, "cached_input": 0, "output": 0}
    t0 = time.perf_counter()
    for i in range(0, len(images), batch):
        if max_usd is not None and spent + last > max_usd:  # le prochain lot coûterait autant que le dernier
            break
        chunk = images[i:i + batch]
        content = [{"type": "text", "text": PROMPT}]
        for im in chunk:
            content += [{"type": "text", "text": f"Image n°{im['id']}"},
                        {"type": "image_url", "image_url": _data_url(im["path"], max_side)}]
        kw = {} if reasoning is None else {"reasoning_effort": reasoning}
        r = client.chat.complete(model=model, messages=[{"role": "user", "content": content}],
                                 response_format={"type": "json_object"}, max_tokens=4000, **kw)
        used = _tokens(r.usage)
        for k in tokens:
            tokens[k] += used[k]
        last = cost(model, used) or 0.0
        spent += last
        try:
            answer = json.loads(r.choices[0].message.content).get("images", [])
        except (json.JSONDecodeError, AttributeError):
            answer = []
        got = {str(x.get("id")): x for x in answer if isinstance(x, dict)}
        for im in chunk:
            x = got.get(str(im["id"]), {})
            labels.append({"id": im["id"], "ajout": bool(x.get("ajout")), "types": x.get("types") or [],
                           "description": x.get("description", ""), "manquant": not x})
    return {"labels": labels, "model": model, "tokens": tokens, "cost_usd": round(cost(model, tokens), 4)
            if model in PRICES else None, "duration_s": round(time.perf_counter() - t0, 1),
            "complete": len(labels) == len(images)}


def annotate_video(result: dict, job: Path, client=None) -> dict:
    """Étiquettes des images d'une analyse de l'interface (result.json d'un job)."""
    frames = result["frames"]
    images = [{"id": i, "path": job / "frames" / f["url"].rsplit("/", 1)[1]} for i, f in enumerate(frames)]
    out = annotate(images, client)
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
    """Correction manuelle de l'image i (None = revenir à l'étiquette du LLM)."""
    path = REAL_DIR / Path(job_id).name / "labels.json"
    labels = json.loads(path.read_text(encoding="utf-8"))
    labels["frames"][i]["manuel"] = value
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    return labels


def load(job_id: str) -> dict | None:
    path = REAL_DIR / Path(job_id).name / "labels.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
