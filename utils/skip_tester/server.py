"""Test du saut des images quasi identiques (frames.skip_refs) : on choisit une vidéo du dataset ou on
en envoie une, et on voit les images que l'annotation enverrait aux annotateurs, chaque image sautée
rangée derrière l'image gardée à laquelle elle a été jugée identique. Le seuil se règle dans la page.

Les vidéos du dataset ont été annotées image par image avant ce mécanisme : on y voit les images
sautées dont l'étiquette diffère de celle de leur référence (l'information que le saut aurait
perdue). Aucun appel payant.

    .venv/Scripts/python.exe utils/skip_tester/server.py [--port 8810]

Vidéos envoyées et leurs images dans utils/skip_tester/_jobs/, effaçables à tout moment.
"""

import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from model_training_pipeline.features import embed_images, get_features  # noqa: E402
from model_training_pipeline.frames import (FPS, LONG_SIDE, SKIP_SIMILAR, extract, frames_of,  # noqa: E402
                                            load_votes, skip_refs)
from utils import dev_reload  # noqa: E402
from video_source.config import FRAMES_DIR, SAMPLES_DIR, ffmpeg_bin  # noqa: E402
from video_source.youtube import load_meta  # noqa: E402

JOBS = HERE / "_jobs"
EXTRACTOR = "clip_b32"  # celui de frames.to_annotate
_gpu = Lock()  # un calcul d'embeddings à la fois
_cache: dict[str, dict] = {}  # id -> {"frames": [url], "X": embeddings, "labels": [1|0|None]}


def _dataset(sid: str) -> dict:
    sdir = SAMPLES_DIR / sid
    frames = frames_of(sdir)
    with _gpu:
        X = get_features([{"sample_id": sid, "frame": f.name} for f in frames], EXTRACTOR, "full")
    votes = load_votes(sdir)["frames"]
    return {"frames": [f"/img/ds/{sid}/{f.name}" for f in frames], "X": X,
            "labels": [votes.get(f.name, {}).get("label") for f in frames]}


def _upload(job: str) -> dict:
    d = JOBS / job
    frames = sorted((d / "img").glob("*.jpg"))
    emb = d / f"{EXTRACTOR}.npy"
    if not emb.exists():
        import torch

        with _gpu, torch.inference_mode():
            np.save(emb, embed_images(frames, EXTRACTOR, "full"))
    return {"frames": [f"/img/up/{job}/{f.name}" for f in frames], "X": np.load(emb),
            "labels": [None] * len(frames)}


def _load(vid: str) -> dict:
    if vid not in _cache:
        kind, _, name = vid.partition(":")
        name = Path(name).name
        if kind == "ds" and (SAMPLES_DIR / name / "meta.json").exists():
            _cache[vid] = _dataset(name)
        elif kind == "up" and (JOBS / name / "img").exists():
            _cache[vid] = _upload(name)
        else:
            raise FileNotFoundError(vid)
    return _cache[vid]


def _duration(video: Path) -> float | None:
    """Durée en secondes, lue dans la sortie de ffmpeg -i."""
    err = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(video)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _differs(labels: list, refs: list[int]) -> tuple[int, int]:
    """(images sautées dont l'étiquette et celle de la référence sont connues, dont elles diffèrent)."""
    pairs = [(labels[i], labels[r]) for i, r in enumerate(refs) if r != i
             and labels[i] is not None and labels[r] is not None]
    return len(pairs), sum(a != b for a, b in pairs)


def simulate(vid: str, threshold: float) -> dict:
    """Ce que l'annotation enverrait d'une vidéo encore jamais annotée (aucun vote pris en compte)."""
    v = _load(vid)
    X = _unit(v["X"])
    refs = skip_refs(v["X"], None, threshold)
    frames, last_kept = [], None
    for i, r in enumerate(refs):
        kept = r == i
        # gardée : cosinus avec la gardée précédente (resté sous le seuil) ; sautée : avec sa référence
        other = last_kept if kept else r
        frames.append({"i": i, "url": v["frames"][i], "kept": kept, "ref": r, "label": v["labels"][i],
                       "cos": None if other is None else round(float(X[i] @ X[other]), 4),
                       "cos_prev": None if i == 0 else round(float(X[i] @ X[i - 1]), 4)})
        if kept:
            last_kept = i
    known, diff = _differs(v["labels"], refs)
    return {"id": vid, "seuil": threshold, "images": len(refs), "envoyees": sum(f["kept"] for f in frames),
            "etiquetee": any(lab is not None for lab in v["labels"]),
            "sautees_comparables": known, "sautees_differentes": diff, "frames": frames}


def videos() -> list[dict]:
    out = []
    for d in sorted(SAMPLES_DIR.glob("yt_*")):
        if (d / "meta.json").exists():
            m = load_meta(d)
            out.append({"id": f"ds:{d.name}", "title": m.get("title"), "channel": m.get("channel"),
                        "duration": m["video"]["duration"], "kind": "dataset"})
    for d in sorted(JOBS.glob("*"), reverse=True) if JOBS.exists() else []:
        if (d / "img").exists():
            name = (d / "name.txt").read_text(encoding="utf-8") if (d / "name.txt").exists() else d.name
            out.append({"id": f"up:{d.name}", "title": name, "channel": None,
                        "duration": len(list((d / "img").glob("*.jpg"))), "kind": "envoyee"})
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, fmt, *args):
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if dev_reload.enabled():
            if u.path == "/__dev":
                return self._json(dev_reload.status(HERE / "index.html"))
            if u.path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, dev_reload.page(HERE / "index.html"), "text/html; charset=utf-8")
        try:
            if u.path == "/api/videos":
                return self._json({"videos": videos(), "seuil_pipeline": SKIP_SIMILAR})
            if u.path == "/api/simulate":
                return self._json(simulate(q.get("id", [""])[0], float(q.get("seuil", [SKIP_SIMILAR])[0])))
        except FileNotFoundError:
            return self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
        if u.path.startswith("/img/"):
            parts = unquote(u.path[5:]).split("/")
            if len(parts) != 3 or parts[0] not in ("ds", "up"):
                return self.send_error(HTTPStatus.NOT_FOUND)
            base = FRAMES_DIR / "img" if parts[0] == "ds" else JOBS
            f = base / Path(parts[1]).name / ("" if parts[0] == "ds" else "img") / Path(parts[2]).name
            if not f.exists():
                return self.send_error(HTTPStatus.NOT_FOUND)
            return self._send(HTTPStatus.OK, f.read_bytes(), "image/jpeg")
        if u.path.endswith(".py") or u.path.startswith("/_jobs"):
            return self.send_error(HTTPStatus.NOT_FOUND)
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/upload":
            return self.send_error(HTTPStatus.NOT_FOUND)
        size = int(self.headers.get("Content-Length") or 0)
        if not size:
            return self._json({"error": "vidéo vide"}, HTTPStatus.BAD_REQUEST)
        name = Path(unquote(self.headers.get("X-Filename") or "video.mp4")).name
        job = JOBS / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job.mkdir(parents=True)
        video = job / ("video" + (Path(name).suffix.lower() or ".mp4"))
        with open(video, "wb") as fh:
            left = size
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                fh.write(chunk)
                left -= len(chunk)
        (job / "name.txt").write_text(name, encoding="utf-8")
        try:
            frames = extract(video, job / "img", FPS, LONG_SIDE)
            # comme frames_of : sans les images qu'ffmpeg ajoute après la fin de la vidéo
            duration = _duration(video)
            for i, f in enumerate(frames):
                if duration and i / FPS >= duration:
                    f.unlink()
            return self._json({"id": f"up:{job.name}"})
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    p = argparse.ArgumentParser(prog="utils/skip_tester/server.py")
    p.add_argument("--port", type=int, default=8810)
    a = p.parse_args()
    print(f"test du saut d'images — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
