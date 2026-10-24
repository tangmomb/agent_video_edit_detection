"""Annotation des images en mode batch (API batch d'OpenAI et de Mistral : 50 % moins cher, résultats
en quelques minutes à 24 h). Même requête qu'en appel direct (annotators.chat_body), une ligne par
image et par annotateur manquant ; les votes reçus sont fusionnés dans votes.json comme ceux de
frames.annotate.

    python -m model_training_pipeline batch estimate     # requêtes à envoyer et coût estimé (gratuit)
    python -m model_training_pipeline batch submit --max-usd 1.0
    python -m model_training_pipeline batch status
    python -m model_training_pipeline batch collect      # récupère les batchs finis, puis reconstruit l'index

Suivi des batchs envoyés dans data/batches.json : une image déjà envoyée n'est pas renvoyée tant que
son batch n'est pas récupéré. Une requête en échec (refusée, expirée) est renvoyée au submit suivant.
Effort de raisonnement de chaque modèle : annotators.reasoning.
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
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"jobs": []}


def _save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def _cid(sdir: Path, frame: Path) -> str:
    return f"{sdir.name}/{frame.name}"


def _todo(state: dict, samples: list[Path] | None = None) -> dict[str, list[tuple[Path, Path]]]:
    """{modèle: [(dossier vidéo, image)]} : votes manquants (de toutes les vidéos, ou de samples),
    hors requêtes d'un batch en cours."""
    in_flight = {(j["model"], c) for j in state["jobs"] if not j.get("collected") for c in j["custom_ids"]}
    out = defaultdict(list)
    for sdir in sorted(SAMPLES_DIR.glob("yt_*")) if samples is None else [Path(d) for d in samples]:
        if not (sdir / "meta.json").exists():
            continue
        votes = frames.load_votes(sdir)
        for f in frames.frames_of(sdir):
            for m in frames.missing(votes, f.name):
                if (m, _cid(sdir, f)) not in in_flight:
                    out[m].append((sdir, f))
    return out


def _measured(model: str, state: dict) -> float | None:
    """Coût réel moyen d'une requête du modèle (tarif batch), sur ses batchs déjà récupérés."""
    done = [j for j in state["jobs"] if j["model"] == model and j.get("collected") and j.get("cost_usd")]
    n = sum(j["requests"] for j in done)
    return sum(j["cost_usd"] for j in done) / n if n else None


def _estimate(model: str, items: list[tuple[Path, Path]], state: dict | None = None) -> float:
    """Coût estimé au tarif batch des requêtes d'un modèle : coût réel moyen par requête si le
    modèle a déjà été mesuré, sinon estimation a priori (ordre de grandeur, prudente)."""
    if not items:
        return 0.0
    per_request = _measured(model, state if state is not None else _load())
    if per_request is not None:
        return per_request * len(items)
    w, h = Image.open(items[0][1]).size
    k = min(1.0, annotators.MAX_SIDE / max(w, h))
    cols, rows = -(-round(w * k) // 16), -(-round(h * k) // 16)
    tin = len(items) * (cols * rows + rows + len(annotators.PROMPT) // 3.5)
    return annotators.cost(model, {"input": int(tin), "cached_input": 0,
                                   "output": len(items) * OUT_PER_IMAGE}) * DISCOUNT


def estimate(samples: list[Path] | None = None) -> dict:
    """Requêtes à envoyer par modèle et coût estimé au tarif batch (ordre de grandeur), pour toutes
    les vidéos ou seulement samples."""
    state = _load()
    todo = _todo(state, samples)
    out, total = {}, 0.0
    for model, items in todo.items():
        usd = _estimate(model, items, state)
        out[model] = {"requetes": len(items), "usd_estime": round(usd, 4),
                      "source": "mesuré" if _measured(model, state) is not None else "a priori"}
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


def _send(provider: str, api, model: str, chunk: list[tuple[Path, Path]], reasoning: str | None) -> str:
    """Écrit le fichier JSONL d'un paquet de requêtes, l'envoie et crée le batch ; renvoie son id."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        for sdir, f in chunk:
            line = {"custom_id": _cid(sdir, f),
                    "body": annotators.chat_body(provider, model, f, annotators.MAX_SIDE, reasoning)}
            if provider == "openai":
                line.update(method="POST", url="/v1/chat/completions")
            fh.write(json.dumps(line) + "\n")
        path = Path(fh.name)
    try:
        return _upload_and_create(provider, api, model, path)
    finally:
        path.unlink(missing_ok=True)


def submit(max_usd: float, apis: dict | None = None, samples: list[Path] | None = None) -> dict:
    """Envoie les votes manquants (de toutes les vidéos, ou de samples), par modèle et par paquets de
    CHUNK requêtes ; refuse si le coût estimé dépasse max_usd."""
    est = estimate(samples)
    if est["total_usd_estime"] > max_usd:
        raise ValueError(f"coût estimé {est['total_usd_estime']} $ > plafond {max_usd} $ : rien n'est envoyé")
    state = _load()
    apis = apis if apis is not None else {}
    sent = []
    for model, items in _todo(state, samples).items():
        provider = annotators.PROVIDER[model]
        if provider not in DONE:
            raise ValueError(f"pas de batch pour {provider} ({model})")
        api = apis.setdefault(provider, annotators.client(provider))
        for i in range(0, len(items), CHUNK):
            chunk = items[i:i + CHUNK]
            reasoning = annotators.reasoning(model)
            job_id = _send(provider, api, model, chunk, reasoning)
            job ={"provider": provider, "model": model, "job_id": job_id, "requests": len(chunk),
                   "reasoning": reasoning, "usd_estime": round(_estimate(model, chunk, state), 5),
                   "custom_ids": [_cid(s, f) for s, f in chunk],
                   "submitted": time.strftime("%Y-%m-%d %H:%M"), "status": None, "collected": False}
            state["jobs"].append(job)
            _save(state)  # après chaque envoi : un batch créé n'est jamais oublié
            sent.append({k: job[k] for k in ("provider", "model", "job_id", "requests", "usd_estime")})
    return {"envoyes": sent, "estimation": est}


def _job(provider: str, api, job_id: str):
    return api.batches.retrieve(job_id) if provider == "openai" else api.batch.jobs.get(job_id=job_id)


def status(apis: dict | None = None, job_ids: set[str] | None = None) -> list[dict]:
    """État des batchs (tous, ou seulement job_ids : l'API n'est interrogée que pour eux)."""
    state = _load()
    apis = apis if apis is not None else {}
    out = []
    for j in state["jobs"]:
        if job_ids is not None and j["job_id"] not in job_ids:
            continue
        if not j.get("collected"):
            api = apis.setdefault(j["provider"], annotators.client(j["provider"]))
            j["status"] = str(_job(j["provider"], api, j["job_id"]).status)
        out.append({k: j.get(k) for k in ("model", "job_id", "requests", "submitted", "status", "collected")})
    _save(state)
    return out


def in_flight(job_ids: set[str] | None = None) -> list[dict]:
    """Batchs envoyés pas encore récupérés (tous, ou parmi job_ids)."""
    return [j for j in _load()["jobs"] if not j.get("collected") and (job_ids is None or j["job_id"] in job_ids)]


def wait(job_ids: set[str], every_s: float = 180, apis: dict | None = None) -> None:
    """Attend, sans limite, que les batchs job_ids soient terminés côté API (sans les récupérer) ;
    les API les arrêtent d'elles-mêmes au bout de 24 h (statut expiré)."""
    while True:
        pending = {j["job_id"] for j in status(apis, job_ids) if not j["collected"]
                   and j["status"] not in DONE["openai"] | DONE["mistral"]}
        if not pending:
            return
        time.sleep(every_s)


def _download(provider: str, api, file_id: str | None) -> list[dict]:
    if not file_id:
        return []
    if provider == "openai":
        text = api.files.content(file_id).text
    else:
        r = api.files.download(file_id=file_id)  # réponse en flux : à lire avant d'en prendre le texte
        r.read()
        text = r.text
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _timing(job) -> dict:
    """Fin du batch et durée depuis sa création, d'après les horodatages de l'API (secondes Unix ;
    OpenAI n'a pas de completed_at pour un batch en échec, expiré ou annulé)."""
    ends = [getattr(job, k, None) for k in ("completed_at", "failed_at", "expired_at", "cancelled_at")]
    end, start = next((t for t in ends if isinstance(t, int)), None), getattr(job, "created_at", None)
    if not (isinstance(end, int) and isinstance(start, int)):
        return {}
    return {"finished": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end)),
            "duration_min": round((end - start) / 60, 1)}


def collect(apis: dict | None = None, job_ids: set[str] | None = None) -> dict:
    """Récupère les batchs terminés (tous, ou parmi job_ids), fusionne leurs votes dans votes.json et
    reconstruit l'index."""
    state = _load()
    apis = apis if apis is not None else {}
    report = {"batchs_recuperes": 0, "votes": 0, "echecs": Counter(), "cost_usd": 0.0, "en_cours": 0,
              "jobs": {}}  # jobs : coût réel de chaque batch récupéré
    by_video: dict[str, dict] = {}
    for j in state["jobs"]:
        if j.get("collected") or (job_ids is not None and j["job_id"] not in job_ids):
            continue
        api = apis.setdefault(j["provider"], annotators.client(j["provider"]))
        job = _job(j["provider"], api, j["job_id"])
        j["status"] = str(job.status)
        if j["status"] not in DONE[j["provider"]]:
            report["en_cours"] += 1
            continue
        out_id = job.output_file_id if j["provider"] == "openai" else job.output_file
        err_id = job.error_file_id if j["provider"] == "openai" else job.error_file
        job_cost = 0.0
        for line in _download(j["provider"], api, out_id) + _download(j["provider"], api, err_id):
            sid, frame = line["custom_id"].split("/", 1)
            resp = line.get("response") or {}
            body, code = resp.get("body"), resp.get("status_code", 200)
            if line.get("error") or code != 200 or not isinstance(body, dict) or "choices" not in body:
                err = json.dumps(line.get("error") or body or line)[:300]
                report["echecs"][err[:120]] += 1
                continue  # vote toujours manquant : renvoyé au prochain submit
            text, used = annotators.read_chat(body)
            usd = annotators.cost(j["model"], used) * DISCOUNT
            if sid not in by_video:
                by_video[sid] = frames.load_votes(SAMPLES_DIR / sid)
            frames.add_votes(by_video[sid], frame, {j["model"]: annotators.parse(text)}, usd)
            report["votes"] += 1
            job_cost += usd
        j["collected"] = True
        j["cost_usd"] = round(job_cost, 5)
        j.update(_timing(job))
        report["cost_usd"] += job_cost
        report["batchs_recuperes"] += 1
        report["jobs"][j["job_id"]] = job_cost
    for sid, votes in by_video.items():
        frames.save_votes(SAMPLES_DIR / sid, votes)
    _save(state)  # votes écrits d'abord : un batch marqué récupéré a toujours ses votes sur disque
    if report["batchs_recuperes"]:
        frames.build()
    report["cost_usd"] = round(report["cost_usd"], 4)
    report["echecs"] = dict(report["echecs"])
    return report
