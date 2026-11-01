"""Description des ajouts de montage visibles sur une image, par un LLM de vision en batch : sert à
l'analyse des erreurs d'un modèle (analyze_errors avec analyse_visuelle), pas à l'étiquetage.

Une description ne dépend que de l'image : elle est gardée dans descriptions.json, à côté des images
de la vidéo (img/<sample_id>/), et jamais redemandée. Une analyse ne paie donc que les images pas
encore décrites. Les batchs de description ne passent pas par batches.json (celui de l'annotation) :
envoi, attente et récupération se font dans le même appel ; un appel interrompu perd son batch.
"""

import json
import tempfile
import time
from pathlib import Path

from video_source import annotators

from . import batch
from .frames import FRAMES_DIR

DESCRIBER = "mistral-large-4-0"
PROMPT = """Décris les ajouts de montage visibles sur cette image (tout ce qui ne vient pas de la \
caméra : texte, titre, sous-titres, logo, bandeau, image incrustée, cadre, effet graphique…) : pour \
chacun, sa nature, sa position, sa taille et son contraste, en une phrase courte. S'il n'y en a \
aucun, dis ce qui dans l'image pourrait y faire penser, ou « aucun ».
Réponds uniquement en JSON : {"description": "…"}"""
# Tokens d'une requête (image de 768 px au plus, consigne, réponse) : estimation avant envoi, calée
# sur la première analyse d'exp_008 (329 images, 0,27 $ en batch, 01/11/2026).
TOKENS_IN, TOKENS_OUT = 1900, 150


def _path(sample_id: str) -> Path:
    return FRAMES_DIR / "img" / sample_id / "descriptions.json"


def load(sample_id: str) -> dict[str, str]:
    """{image: description} d'une vidéo."""
    p = _path(sample_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def estimate(n: int) -> float:
    """Coût estimé de n descriptions au tarif batch."""
    return annotators.cost(DESCRIBER, {"input": TOKENS_IN * n, "cached_input": 0,
                                       "output": TOKENS_OUT * n}) * batch.DISCOUNT


def describe(frames: list[str], max_usd: float, every_s: float = 20) -> dict:
    """Décrit les images (chemins relatifs à FRAMES_DIR, img/<sample_id>/<image>) pas encore
    décrites, en un batch, en attendant le résultat. Refuse si le coût estimé dépasse max_usd.
    Renvoie {"descriptions": {chemin: texte}, "cost_usd", "nouvelles", "echecs"}."""
    known = {}
    for f in frames:
        sid, name = f.split("/")[1], f.split("/")[-1]
        d = load(sid)
        if name in d:
            known[f] = d[name]
    todo = [f for f in frames if f not in known]
    report = {"descriptions": known, "cost_usd": 0.0, "nouvelles": 0, "echecs": 0}
    if not todo:
        return report
    if estimate(len(todo)) > max_usd:
        raise ValueError(f"{len(todo)} images à décrire, ≈ {estimate(len(todo)):.2f} $ > {max_usd:.2f} $")
    provider = annotators.PROVIDER[DESCRIBER]
    api = annotators.client(provider)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        for f in todo:
            line = {"custom_id": f, "body": annotators.chat_body(
                provider, DESCRIBER, FRAMES_DIR / f, annotators.MAX_SIDE, None, PROMPT)}
            if provider == "openai":
                line.update(method="POST", url="/v1/chat/completions")
            fh.write(json.dumps(line) + "\n")
        path = Path(fh.name)
    try:
        job_id = batch._upload_and_create(provider, api, DESCRIBER, path)
    finally:
        path.unlink(missing_ok=True)
    while True:
        job = batch._job(provider, api, job_id)
        if str(job.status) in batch.DONE[provider]:
            break
        time.sleep(every_s)
    out_id = job.output_file_id if provider == "openai" else job.output_file
    err_id = job.error_file_id if provider == "openai" else job.error_file
    new: dict[str, dict[str, str]] = {}
    for line in batch._download(provider, api, out_id) + batch._download(provider, api, err_id):
        body = (line.get("response") or {}).get("body")
        if line.get("error") or not isinstance(body, dict) or "choices" not in body:
            report["echecs"] += 1
            continue
        text, used = annotators.read_chat(body)
        report["cost_usd"] += annotators.cost(DESCRIBER, used) * batch.DISCOUNT
        try:
            desc = str(json.loads(text)["description"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            report["echecs"] += 1
            continue
        f = line["custom_id"]
        new.setdefault(f.split("/")[1], {})[f.split("/")[-1]] = desc
        report["descriptions"][f] = desc
        report["nouvelles"] += 1
    for sid, d in new.items():
        _path(sid).write_text(json.dumps({**load(sid), **d}, ensure_ascii=False), encoding="utf-8")
    report["cost_usd"] = round(report["cost_usd"], 4)
    return report
