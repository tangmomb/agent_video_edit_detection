"""Banc des annotateurs (Mistral, OpenAI, Anthropic) : on charge une vidéo, chaque modèle choisi dit
en JSON quelles images (1 par seconde) contiennent un ajout de montage, et on compare leurs réponses,
leur coût et leur durée pour choisir le prochain modèle d'annotation. Outil autonome : il n'est
branché ni au dataset, ni au classifieur, ni aux agents.

    .venv/Scripts/python.exe utils/llm_bench/server.py [--port 8790]

Chaque vidéo chargée est rangée dans utils/llm_bench/_videos/<id>/ (vidéo, images, et une réponse
JSON par modèle dans runs/). Chaque lancement de modèle est un appel payant à l'API de son
fournisseur (clés MISTRAL_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY du .env).
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from utils import dev_reload  # noqa: E402

load_dotenv(ROOT / ".env")
VIDEOS = HERE / "_videos"

# Modèles comparés (docs.mistral.ai/models, developers.openai.com/api/docs/pricing,
# platform.claude.com, octobre 2026) : identifiant d'API, nom, lecture d'images, tarif en $ par
# million de tokens, fournisseur.
MODELS = [
    ("mistral-large-4-0", "Mistral Large 4", True, {"input": 1.36, "cached_input": 0.14, "output": 4.18}, "mistral"),
    ("mistral-large-2512", "Mistral Large 3", True, {"input": 0.50, "output": 1.50}, "mistral"),
    ("mistral-small-2603", "Mistral Small 4", True, {"input": 0.15, "output": 0.60}, "mistral"),
    ("ministral-14b-2512", "Ministral 3 14B", True, {"input": 0.20, "output": 0.20}, "mistral"),
    ("zai-glm-5-3-flash", "Z.ai GLM 5.3 Flash", True, {"input": 0.15, "cached_input": 0.03, "output": 0.50}, "mistral"),
    ("gpt-6-luna", "GPT-6 Luna", True, {"input": 0.10, "cached_input": 0.01, "output": 0.50}, "openai"),
    ("gpt-5.6-luna", "GPT-5.6 Luna", True, {"input": 0.20, "cached_input": 0.02, "output": 1.20}, "openai"),
    ("claude-haiku-5-5", "Claude Haiku 5.5", True, {"input": 0.10, "output": 0.50}, "anthropic"),
]
PRICES = {m[0]: m[3] for m in MODELS}
PROVIDER = {m[0]: m[4] for m in MODELS}
KEYS = {"mistral": "MISTRAL_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
REASONING = "medium"  # même effort de raisonnement pour tous les modèles qui raisonnent

PROMPT = """Dis si cette image est brute de la caméra ou non.
Réponds true dès que quelque chose a été ajouté au montage : habillage, texte, titre, sous-titres,
photo ou image incrustée, logo, cadre, effet graphique ou tout autre élément qui ne vient pas de la caméra.
Réponds false seulement si l'image est telle que la caméra l'a filmée.
Réponds uniquement en JSON : {"ajout": true|false}"""


def client(provider: str):
    key = os.environ.get(KEYS[provider])
    if not key:
        raise ValueError(f"{KEYS[provider]} absente du .env : impossible d'interroger {provider}")
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=key)
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=key)
    from mistralai.client import Mistral
    return Mistral(api_key=key)


def _ask(provider: str, api, model: str, path: Path, max_side: int, reasoning: str | None) -> tuple[str, dict]:
    """Une requête (consigne + une image) ; renvoie le texte de la réponse et les tokens consommés.
    reasoning=None : sans effort de raisonnement imposé."""
    url = _data_url(path, max_side)
    if provider == "anthropic":
        # Pas de fallbacks : on compare des modèles, la réponse doit venir du modèle demandé.
        r = api.messages.create(
            model=model, max_tokens=8000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": url.split(",", 1)[1]}},
                {"type": "text", "text": PROMPT}]}],
            output_config={"format": {"type": "json_schema", "schema": {
                "type": "object", "properties": {"ajout": {"type": "boolean"}},
                "required": ["ajout"], "additionalProperties": False}},
                **({"effort": reasoning} if reasoning else {})})
        u = r.usage
        cached, created = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        text = "" if r.stop_reason == "refusal" else "".join(b.text for b in r.content if b.type == "text")
        return text, {"input": u.input_tokens + cached + created, "cached_input": cached, "output": u.output_tokens}
    if provider == "openai":
        content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]
        req = {"model": model, "messages": [{"role": "user", "content": content}],
               "response_format": {"type": "json_object"}, "max_completion_tokens": 8000}
        r = api.chat.completions.create(**req, **({"reasoning_effort": reasoning} if reasoning else {}))
    else:
        content = [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": url}]
        req = {"model": model, "messages": [{"role": "user", "content": content}],
               "response_format": {"type": "json_object"}, "max_tokens": 8000}
        r = api.chat.complete(**req, **({"reasoning_effort": reasoning} if reasoning else {}))
    return _text(r.choices[0].message.content), _usage(r.usage)


def extract(video: Path, out: Path) -> list[Path]:
    """Une image par seconde, 640 px au plus grand côté."""
    import imageio_ffmpeg
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(video), "-vf",
                    "fps=1,scale='if(gt(iw,ih),640,-2)':'if(gt(iw,ih),-2,640)'", "-q:v", "3",
                    str(out / "%05d.jpg")], check=True)
    return sorted(out.glob("*.jpg"))


def frames_of(vid: str) -> list[dict]:
    d = VIDEOS / Path(vid).name / "frames"
    return [{"i": i, "t": i, "url": f"/_videos/{d.parent.name}/frames/{p.name}"}
            for i, p in enumerate(sorted(d.glob("*.jpg")))]


def _data_url(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _cost(model: str, tokens: dict) -> float:
    p = PRICES[model]
    fresh = tokens["input"] - tokens["cached_input"]
    return (fresh * p["input"] + tokens["cached_input"] * p.get("cached_input", p["input"])
            + tokens["output"] * p["output"]) / 1e6


def _text(content) -> str:
    """Texte de la réponse : avec le raisonnement, content est une liste de morceaux (réflexion puis
    texte) ; on ne garde que le texte."""
    if isinstance(content, str):
        return content
    get = lambda c, k: c.get(k) if isinstance(c, dict) else getattr(c, k, None)  # noqa: E731
    return "".join(get(c, "text") or "" for c in content or [] if get(c, "type") == "text")


def _usage(u) -> dict:
    details = getattr(u, "prompt_tokens_details", None)
    cached = (details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", 0)) or 0
    return {"input": u.prompt_tokens or 0, "cached_input": cached, "output": u.completion_tokens or 0}


def run(cfg: dict, api=None) -> dict:
    """Fait annoter toutes les images de la vidéo par un modèle ; enregistre et renvoie sa réponse."""
    model = cfg["model"]
    if not any(m[0] == model and m[2] for m in MODELS):
        raise ValueError(f"{model} ne lit pas les images (ou n'est pas dans la liste) : il ne peut pas annoter")
    d = VIDEOS / Path(cfg["video"]).name
    paths = sorted((d / "frames").glob("*.jpg"))
    max_side = int(cfg.get("max_side", 768))
    max_usd, reasoning = float(cfg.get("max_usd", 0.5)), REASONING
    provider = PROVIDER[model]
    api = api or client(provider)
    labels, raw, spent, last, refus = [], [], 0.0, 0.0, None
    tokens = {"input": 0, "cached_input": 0, "output": 0}
    t0 = time.perf_counter()
    for i, path in enumerate(paths):  # une requête par image
        if spent + last > max_usd:  # la prochaine image coûterait à peu près autant que la dernière
            break
        try:
            text, used = _ask(provider, api, model, path, max_side, reasoning)
        except Exception as e:
            if not reasoning or not any(w in str(e).lower() for w in ("reason", "effort")):
                raise
            reasoning, refus = None, str(e)[:300]  # refusé : on continue sans, et on garde la réponse de l'API
            text, used = _ask(provider, api, model, path, max_side, None)
        for k in tokens:
            tokens[k] += used[k]
        last = _cost(model, used)
        spent += last
        raw.append(text)
        try:
            answer = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
        except json.JSONDecodeError:
            answer = None
        ok = isinstance(answer, dict) and "ajout" in answer
        labels.append({"i": i, "ajout": ok and bool(answer["ajout"]), "manquant": not ok})
    config = {"max_side": max_side, "batch": 1, "max_usd": max_usd,
              "reasoning": reasoning or f"« {REASONING} » refusé, lancé sans", "refus_api": refus}
    rec = {"model": model, "config": config, "images_total": len(paths),
           "frames": labels, "tokens": tokens, "cost_usd": round(_cost(model, tokens), 4),
           "duration_s": round(time.perf_counter() - t0, 1), "complete": len(labels) == len(paths),
           "raw": raw, "date": time.strftime("%Y-%m-%d %H:%M")}
    (d / "runs").mkdir(exist_ok=True)
    (d / "runs" / f"{model}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


# Estimation a priori, avant tout lancement : une image coûte à peu près un token par carreau de
# 16 × 16 px (plus un par ligne), la consigne environ 1 token pour 3,5 caractères ; la réponse
# environ 15 tokens par image, plus la réflexion du raisonnement « medium » (ordre de grandeur).
OUT_PER_IMAGE = 15 + 150


def _image_tokens(w: int, h: int, max_side: int) -> int:
    k = min(1.0, max_side / max(w, h))
    cols, rows = -(-round(w * k) // 16), -(-round(h * k) // 16)
    return cols * rows + rows


def estimate(vid: str, max_side: int) -> dict:
    """Coût estimé par modèle pour annoter toute la vidéo. Quand le modèle a déjà été lancé à la même
    taille d'image (sur n'importe quelle vidéo), on prend ses tokens réels par image."""
    paths = sorted((VIDEOS / Path(vid).name / "frames").glob("*.jpg"))
    if not paths:
        return {}
    w, h = Image.open(paths[0]).size
    n = len(paths)
    base_in = n * (_image_tokens(w, h, max_side) + len(PROMPT) // 3.5)
    seen: dict[str, list[tuple[int, int, int]]] = {}
    for p in VIDEOS.glob("*/runs/*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["config"].get("max_side") == max_side and rec["frames"]:
            seen.setdefault(rec["model"], []).append((len(rec["frames"]), rec["tokens"]["input"], rec["tokens"]["output"]))
    out = {}
    for model, *_ in MODELS:
        runs = seen.get(model)
        if runs:
            imgs = sum(r[0] for r in runs)
            tin, tout, src = n * sum(r[1] for r in runs) / imgs, n * sum(r[2] for r in runs) / imgs, "mesuré"
        else:
            tin, tout, src = base_in, n * OUT_PER_IMAGE, "estimé"
        out[model] = {"usd": round(_cost(model, {"input": int(tin), "cached_input": 0, "output": int(tout)}), 4),
                      "source": src}
    return out


def video(vid: str) -> dict:
    d = VIDEOS / Path(vid).name
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((d / "runs").glob("*.json"))] \
        if (d / "runs").exists() else []
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return {**meta, "frames": frames_of(vid), "runs": runs}


def videos() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(VIDEOS.glob("*/meta.json"), reverse=True)] \
        if VIDEOS.exists() else []


class Handler(SimpleHTTPRequestHandler):
    api = None  # remplacé par un faux client dans les tests

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("[bench] " + fmt % args + "\n")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        if dev_reload.enabled():  # lancé par utils/dev_reload.py : la page se recharge toute seule
            if u.path == "/__dev":
                return self._json(HTTPStatus.OK, dev_reload.status(HERE / "index.html"))
            if u.path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, dev_reload.page(HERE / "index.html"), "text/html; charset=utf-8")
        if u.path == "/api/models":
            return self._json(HTTPStatus.OK, {"models": [{"id": i, "nom": n, "vision": v, "prix": p, "fournisseur": f,
                                                          "cle": KEYS[f] if not os.environ.get(KEYS[f]) else None}
                                                         for i, n, v, p, f in MODELS],
                                              "prompt": PROMPT, "reasoning": REASONING})
        if u.path == "/api/estimate":
            q = parse_qs(u.query)
            return self._json(HTTPStatus.OK, estimate(q.get("video", [""])[0], int(q.get("max_side", ["768"])[0])))
        if u.path == "/api/videos":
            return self._json(HTTPStatus.OK, videos())
        if u.path == "/api/video":
            vid = parse_qs(u.query).get("id", [""])[0]
            if not (VIDEOS / Path(vid).name / "meta.json").exists():
                return self.send_error(HTTPStatus.NOT_FOUND)
            return self._json(HTTPStatus.OK, video(vid))
        if u.path.endswith(".py") or "/runs/" in u.path:
            return self.send_error(HTTPStatus.NOT_FOUND)
        return super().do_GET()

    def do_POST(self):
        try:
            if self.path == "/api/upload":
                return self._upload()
            if self.path == "/api/run":
                cfg = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                return self._json(HTTPStatus.OK, run(cfg, api=self.api))
            return self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as e:
            return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(e).__name__}: {e}"})

    def _upload(self):
        size = int(self.headers.get("Content-Length") or 0)
        if not size:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "vidéo vide"})
        from urllib.parse import unquote
        name = Path(unquote(self.headers.get("X-Filename") or "video.mp4")).name
        vid = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        d = VIDEOS / vid
        d.mkdir(parents=True)
        path = d / ("video" + (Path(name).suffix.lower() or ".mp4"))
        with open(path, "wb") as fh:
            left = size
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                fh.write(chunk)
                left -= len(chunk)
        n = len(extract(path, d / "frames"))
        if not n:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "aucune image extraite : fichier vidéo illisible ?"})
        (d / "meta.json").write_text(json.dumps({"id": vid, "nom": name, "images": n,
                                                 "date": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False),
                                     encoding="utf-8")
        return self._json(HTTPStatus.OK, video(vid))


def main() -> None:
    p = argparse.ArgumentParser(prog="utils/llm_bench/server.py")
    p.add_argument("--port", type=int, default=8790)
    a = p.parse_args()
    print(f"banc des annotateurs — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
