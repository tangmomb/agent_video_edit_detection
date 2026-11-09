"""Banc des annotateurs (Mistral, OpenAI, Anthropic) : on charge une vidéo, chaque modèle choisi dit
en JSON quelles images (1 par seconde) contiennent un ajout de montage, et on compare leurs réponses,
leur coût et leur durée pour choisir le prochain modèle d'annotation. Outil autonome : il n'est
branché ni au dataset, ni au classifieur, ni aux agents.

    .venv/Scripts/python.exe interface/llm_bench/server.py [--port 8790]

Chaque vidéo chargée est rangée dans interface/llm_bench/_videos/<id>/ (vidéo, images, et une réponse
JSON par modèle dans runs/). Chaque lancement de modèle est un appel payant à l'API de son
fournisseur (clés MISTRAL_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY du .env).

En ligne (requête passée par Caddy), la page est publique et bornée : seuls les annotateurs du
dataset, sur ses propres vidéos, une fois par modèle et par vidéo, avec un plafond par lancement et un
budget en dollars par jour pour tout le monde (interface/quota.py). En local, rien ne change.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from interface import dev_reload  # noqa: E402
from interface.quota import Quota, Rejected  # noqa: E402
from video_source.annotators import (ANNOTATORS, KEYS, MODELS, PROMPT, PROVIDER, REASONING, REASONING_BY_MODEL,  # noqa: E402
                                     ask, client, parse, reasoning)
from video_source.annotators import cost as _cost  # noqa: E402

VIDEOS = HERE / "_videos"
# En ligne : une vidéo de 3 min coûte environ 0,06 à 0,09 $ par annotateur (180 images, 768 px).
QUOTA = Quota(VIDEOS, per_ip=None, per_day=3, max_mb=500, max_s=180)
PUBLIC_MODELS = ANNOTATORS
USD_PER_RUN = 0.25  # plafond d'un lancement (un modèle sur une vidéo)
USD_PER_DAY = 3.0  # pour tout le monde
PUBLIC_MAX_SIDE = 768


def owner(vid: str) -> str | None:
    f = VIDEOS / Path(vid).name / "owner.txt"
    return f.read_text(encoding="utf-8") if f.exists() else None


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
    return [{"i": i, "t": i, "url": f"_videos/{d.parent.name}/frames/{p.name}"}
            for i, p in enumerate(sorted(d.glob("*.jpg")))]


def run(cfg: dict, api=None) -> dict:
    """Fait annoter toutes les images de la vidéo par un modèle ; enregistre et renvoie sa réponse."""
    model = cfg["model"]
    if not any(m[0] == model and m[2] for m in MODELS):
        raise ValueError(f"{model} ne lit pas les images (ou n'est pas dans la liste) : il ne peut pas annoter")
    d = VIDEOS / Path(cfg["video"]).name
    paths = sorted((d / "frames").glob("*.jpg"))
    max_side = int(cfg.get("max_side", 768))
    max_usd, effort = float(cfg.get("max_usd", 0.5)), reasoning(model)
    provider = PROVIDER[model]
    api = api or client(provider)
    labels, raw, spent, last = [], [], 0.0, 0.0
    tokens = {"input": 0, "cached_input": 0, "output": 0}
    t0 = time.perf_counter()
    for i, path in enumerate(paths):  # une requête par image
        if spent + last > max_usd:  # la prochaine image coûterait à peu près autant que la dernière
            break
        text, used = ask(provider, api, model, path, max_side, effort)
        for k in tokens:
            tokens[k] += used[k]
        last = _cost(model, used)
        spent += last
        raw.append(text)
        answer = parse(text)
        labels.append({"i": i, "ajout": bool(answer), "manquant": answer is None})
    config = {"max_side": max_side, "batch": 1, "max_usd": max_usd,
              "reasoning": effort or "par défaut"}
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


def videos(ip: str | None) -> list[dict]:
    """Les vidéos chargées depuis cette adresse (toutes en local)."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(VIDEOS.glob("*/meta.json"), reverse=True)
            if ip is None or owner(p.parent.name) == ip] if VIDEOS.exists() else []


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
        if dev_reload.enabled():  # lancé par interface/dev_reload.py : la page se recharge toute seule
            if u.path == "/__dev":
                return self._json(HTTPStatus.OK, dev_reload.status(HERE / "index.html"))
            if u.path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, dev_reload.page(HERE / "index.html"), "text/html; charset=utf-8")
        ip = QUOTA.client(self)
        if u.path == "/api/models":
            public = None if ip is None else {
                **QUOTA.limits(), "usd_par_lancement": USD_PER_RUN, "usd_jour": USD_PER_DAY,
                "usd_restant": round(QUOTA.usd_left(USD_PER_DAY), 2), "max_side": PUBLIC_MAX_SIDE}
            return self._json(HTTPStatus.OK, {"models": [{"id": i, "nom": n, "vision": v, "prix": p, "fournisseur": f,
                                                          "cle": KEYS[f] if not os.environ.get(KEYS[f]) else None}
                                                         for i, n, v, p, f in MODELS
                                                         if ip is None or i in PUBLIC_MODELS],
                                              "prompt": PROMPT, "reasoning": REASONING,
                                              "reasoning_by_model": {m: r for m, r in REASONING_BY_MODEL.items()
                                                                     if ip is None or m in PUBLIC_MODELS},
                                              "public": public})
        if u.path == "/api/estimate":
            q = parse_qs(u.query)
            return self._json(HTTPStatus.OK, estimate(q.get("video", [""])[0], int(q.get("max_side", ["768"])[0])))
        if u.path == "/api/videos":
            return self._json(HTTPStatus.OK, videos(ip))
        if u.path == "/api/video":
            vid = parse_qs(u.query).get("id", [""])[0]
            if not (VIDEOS / Path(vid).name / "meta.json").exists() or (ip is not None and owner(vid) != ip):
                return self.send_error(HTTPStatus.NOT_FOUND)
            return self._json(HTTPStatus.OK, video(vid))
        if u.path.endswith((".py", "owner.txt", "quota.json")) or "/runs/" in u.path:  # IP des visiteurs
            return self.send_error(HTTPStatus.NOT_FOUND)
        return super().do_GET()

    def do_POST(self):
        try:
            if self.path == "/api/upload":
                return self._upload()
            if self.path == "/api/run":
                cfg = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                ip = QUOTA.client(self)
                if ip is None:
                    return self._json(HTTPStatus.OK, run(cfg, api=self.api))
                return self._public_run(cfg, ip)
            return self.send_error(HTTPStatus.NOT_FOUND)
        except Rejected as e:
            return self._json(e.status, {"error": str(e)})
        except Exception as e:
            return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(e).__name__}: {e}"})

    def _public_run(self, cfg: dict, ip: str):
        """Un lancement depuis la page publique : bornes de quota.py, puis la dépense réelle comptée."""
        model, vid = cfg.get("model"), Path(str(cfg.get("video", ""))).name
        if model not in PUBLIC_MODELS:
            raise Rejected("En ligne, seuls les annotateurs du dataset peuvent être interrogés.", HTTPStatus.FORBIDDEN)
        if not vid or owner(vid) != ip:
            raise Rejected("Vidéo introuvable.", HTTPStatus.NOT_FOUND)
        if (VIDEOS / vid / "runs" / f"{model}.json").exists():
            raise Rejected("Ce modèle a déjà répondu sur cette vidéo.", HTTPStatus.CONFLICT)
        cfg = {"model": model, "video": vid,
               "max_side": min(int(cfg.get("max_side", PUBLIC_MAX_SIDE)), PUBLIC_MAX_SIDE),
               "max_usd": min(float(cfg.get("max_usd", USD_PER_RUN)), USD_PER_RUN)}
        QUOTA.reserve_usd(cfg["max_usd"], USD_PER_DAY)
        spent = cfg["max_usd"]  # compté en entier si le lancement plante en route
        try:
            rec = run(cfg, api=self.api)
            spent = rec["cost_usd"]
            return self._json(HTTPStatus.OK, rec)
        finally:
            QUOTA.settle_usd(cfg["max_usd"], spent)

    def _upload(self):
        size = int(self.headers.get("Content-Length") or 0)
        if not size:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "vidéo vide"})
        QUOTA.cleanup()
        try:
            ip = QUOTA.admit(self)
        except Rejected as e:
            self.close_connection = True  # le corps n'est pas lu
            return self._json(e.status, {"error": str(e)})
        from urllib.parse import unquote
        name = Path(unquote(self.headers.get("X-Filename") or "video.mp4")).name
        vid = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        d = VIDEOS / vid
        try:
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
            QUOTA.check_video(ip, path)
            n = len(extract(path, d / "frames"))
            if not n:
                QUOTA.refund(ip)
                shutil.rmtree(d, ignore_errors=True)
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "aucune image extraite : fichier vidéo illisible ?"})
            if ip:
                (d / "owner.txt").write_text(ip, encoding="utf-8")
            (d / "meta.json").write_text(json.dumps({"id": vid, "nom": name, "images": n,
                                                     "date": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False),
                                         encoding="utf-8")
            return self._json(HTTPStatus.OK, video(vid))
        except Rejected as e:
            shutil.rmtree(d, ignore_errors=True)
            return self._json(e.status, {"error": str(e)})
        except Exception:
            QUOTA.refund(ip)
            raise
        finally:
            QUOTA.release()

def main() -> None:
    p = argparse.ArgumentParser(prog="interface/llm_bench/server.py")
    p.add_argument("--port", type=int, default=8790)
    a = p.parse_args()
    print(f"banc des annotateurs — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
