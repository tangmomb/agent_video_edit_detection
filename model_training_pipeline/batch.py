"""Annotation des images en mode batch (API batch d'OpenAI et de Mistral : 50 % moins cher, résultats
en quelques minutes à 24 h). Même requête qu'en appel direct (annotators.chat_body), une ligne par
image et par annotateur manquant ; les votes reçus sont fusionnés dans votes.json comme ceux de
frames.annotate.

    python -m model_training_pipeline batch estimate     # requêtes à envoyer et coût estimé (gratuit)
    python -m model_training_pipeline batch submit --max-usd 1.0
    python -m model_training_pipeline batch status
    python -m model_training_pipeline batch collect      # récupère les batchs finis, puis reconstruit l'index

Suivi des batchs envoyés dans data/batches.json : une image déjà envoyée n'est pas renvoyée tant que
son batch n'est pas récupéré. Une requête en échec (refusée, expirée) est renvoyée au submit suivant ;
si l'API a refusé l'effort de raisonnement, le modèle est noté et renvoyé sans.
"""

import json
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from video_source import annotators
from video_source.config import DATA_DIR, SAMPLES_DIR

from . import frames

STATE = DATA_DIR / "batches.json"
DISCOUNT = 0.5  # remise batch d'OpenAI et de Mistral (octobre 2026)
CHUNK = 500  # requêtes par batch : fichiers de quelques dizaines de Mo (images en base64)
DONE = {"openai": {"completed", "failed", "expired", "cancelled"},
        "mistral": {"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}}
# Estimation a priori des tokens (avant toute mesure) : image ~ un token par carreau de 16 px, consigne
# ~ 1 token pour 3,5 caractères, réponse ~ 15 tokens + la réflexion du raisonnement « medium ».
OUT_PER_IMAGE = 15 + 150


def _load() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"jobs": [], "no_reasoning": []}


def _save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def _cid(sdir: Path, frame: Path) -> str:
    return f"{sdir.name}/{frame.name}"


def _todo(state: dict) -> dict[str, list[tuple[Path, Path]]]:
    """{modèle: [(dossier vidéo, image)]} : votes manquants, hors requêtes d'un batch en cours."""
    in_flight = {(j["model"], c) for j in state["jobs"] if not j.get("collected") for c in j["custom_ids"]}
    out = defaultdict(list)
    for sdir in sorted(SAMPLES_DIR.glob("yt_*")):
        if not (sdir / "meta.json").exists():
            continue
        votes = frames.load_votes(sdir)
        for f in frames.frames_of(sdir):
            for m in frames.missing(votes, f.name):
                if (m, _cid(sdir, f)) not in in_flight:
                    out[m].append((sdir, f))
    return out


def estimate() -> dict:
    """Requêtes à envoyer par modèle et coût estimé au tarif batch (ordre de grandeur)."""
    todo = _todo(_load())
    out, total = {}, 0.0
    for model, items in todo.items():
        w, h = Image.open(items[0][1]).size
        k = min(1.0, annotators.MAX_SIDE / max(w, h))
        cols, rows = -(-round(w * k) // 16), -(-round(h * k) // 16)
        tin = len(items) * (cols * rows + rows + len(annotators.PROMPT) // 3.5)
        usd = annotators.cost(model, {"input": int(tin), "cached_input": 0,
                                      "output": len(items) * OUT_PER_IMAGE}) * DISCOUNT
        out[model] = {"requetes": len(items), "usd_estime": round(usd, 4)}
        total += usd
    return {"par_modele": out, "total_usd_estime": round(total, 4),
            "images": len({c for items in todo.values() for c in items})}


def _upload_and_create(provider: str, api, model: str, path: Path) -> str:
    if provider == "openai":
        with open(path, "rb") as fh:
            file_id = api.files.create(file=fh, purpose="batch").id
        return api.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions",
                                  completion_window="24h").id
    with open(path, "rb") as fh:
        file_id = api.files.upload(file={"file_name": path.name, "content": fh.read()}, purpose="batch").id
    return api.batch.jobs.create(input_files=[file_id], model=model, endpoint="/v1/chat/completions").id


def submit(max_usd: float, apis: dict | None = None) -> dict:
    """Envoie les votes manquants, par modèle et par paquets de CHUNK requêtes ; refuse si le coût
    estimé dépasse max_usd."""
    est = estimate()
    if est["total_usd_estime"] > max_usd:
        raise ValueError(f"coût estimé {est['total_usd_estime']} $ > plafond {max_usd} $ : rien n'est envoyé")
    state = _load()
    apis = apis if apis is not None else {}
    sent = []
    for model, items in _todo(state).items():
        provider = annotators.PROVIDER[model]
        if provider not in DONE:
            raise ValueError(f"pas de batch pour {provider} ({model})")
        api = apis.setdefault(provider, annotators.client(provider))
        reasoning = None if model in state["no_reasoning"] else annotators.REASONING
        for i in range(0, len(items), CHUNK):
            chunk = items[i:i + CHUNK]
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
                for sdir, f in chunk:
                    line = {"custom_id": _cid(sdir, f),
                            "body": annotators.chat_body(provider, model, f, annotators.MAX_SIDE, reasoning)}
                    if provider == "openai":
                        line.update(method="POST", url="/v1/chat/completions")
                    fh.write(json.dumps(line) + "\n")
                path = Path(fh.name)
            try:
                job_id = _upload_and_create(provider, api, model, path)
            finally:
                path.unlink(missing_ok=True)
            job = {"provider": provider, "model": model, "job_id": job_id, "requests": len(chunk),
                   "reasoning": reasoning, "custom_ids": [_cid(s, f) for s, f in chunk],
                   "submitted": time.strftime("%Y-%m-%d %H:%M"), "status": None, "collected": False}
            state["jobs"].append(job)
            _save(state)  # après chaque envoi : un batch créé n'est jamais oublié
            sent.append({k: job[k] for k in ("provider", "model", "job_id", "requests")})
    return {"envoyes": sent, "estimation": est}


def _job(provider: str, api, job_id: str):
    return api.batches.retrieve(job_id) if provider == "openai" else api.batch.jobs.get(job_id=job_id)


def status(apis: dict | None = None) -> list[dict]:
    state = _load()
    apis = apis if apis is not None else {}
    out = []
    for j in state["jobs"]:
        if not j.get("collected"):
            api = apis.setdefault(j["provider"], annotators.client(j["provider"]))
            j["status"] = str(_job(j["provider"], api, j["job_id"]).status)
        out.append({k: j.get(k) for k in ("model", "job_id", "requests", "submitted", "status", "collected")})
    _save(state)
    return out


def _download(provider: str, api, file_id: str | None) -> list[dict]:
    if not file_id:
        return []
    text = api.files.content(file_id).text if provider == "openai" else api.files.download(file_id=file_id).text
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def collect(apis: dict | None = None) -> dict:
    """Récupère les batchs terminés, fusionne leurs votes dans votes.json et reconstruit l'index."""
    state = _load()
    apis = apis if apis is not None else {}
    report = {"batchs_recuperes": 0, "votes": 0, "echecs": Counter(), "cost_usd": 0.0, "en_cours": 0}
    by_video: dict[str, dict] = {}
    for j in state["jobs"]:
        if j.get("collected"):
            continue
        api = apis.setdefault(j["provider"], annotators.client(j["provider"]))
        job = _job(j["provider"], api, j["job_id"])
        j["status"] = str(job.status)
        if j["status"] not in DONE[j["provider"]]:
            report["en_cours"] += 1
            continue
        out_id = job.output_file_id if j["provider"] == "openai" else job.output_file
        err_id = job.error_file_id if j["provider"] == "openai" else job.error_file
        for line in _download(j["provider"], api, out_id) + _download(j["provider"], api, err_id):
            sid, frame = line["custom_id"].split("/", 1)
            resp = line.get("response") or {}
            body, code = resp.get("body"), resp.get("status_code", 200)
            if line.get("error") or code != 200 or not isinstance(body, dict) or "choices" not in body:
                err = json.dumps(line.get("error") or body or line)[:300]
                report["echecs"][err[:120]] += 1
                if j["reasoning"] and annotators.refuses_reasoning(Exception(err)) \
                        and j["model"] not in state["no_reasoning"]:
                    state["no_reasoning"].append(j["model"])
                continue  # vote toujours manquant : renvoyé au prochain submit
            text, used = annotators.read_chat(body)
            usd = annotators.cost(j["model"], used) * DISCOUNT
            if sid not in by_video:
                by_video[sid] = frames.load_votes(SAMPLES_DIR / sid)
            frames.add_votes(by_video[sid], frame, {j["model"]: annotators.parse(text)}, usd)
            report["votes"] += 1
            report["cost_usd"] += usd
        j["collected"] = True
        report["batchs_recuperes"] += 1
    for sid, votes in by_video.items():
        frames.save_votes(SAMPLES_DIR / sid, votes)
    _save(state)  # votes écrits d'abord : un batch marqué récupéré a toujours ses votes sur disque
    if report["batchs_recuperes"]:
        frames.build()
    report["cost_usd"] = round(report["cost_usd"], 4)
    report["echecs"] = dict(report["echecs"])
    return report
