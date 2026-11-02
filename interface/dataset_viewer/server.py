"""Visionneuse du dataset : on choisit une vidéo de data/videos_samples et on voit chaque image
(1 par seconde) avec son étiquette (ajout / brute / sans étiquette) et le vote de chaque annotateur.
Lecture seule, aucun appel payant.

    .venv/Scripts/python.exe interface/dataset_viewer/server.py [--port 8800]
"""

import argparse
import json
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from interface import dev_reload  # noqa: E402
from video_source.annotators import ANNOTATORS  # noqa: E402
from video_source.config import FRAMES_DIR, SAMPLES_DIR  # noqa: E402
from video_source.youtube import load_meta  # noqa: E402


def _votes(d: Path) -> dict:
    p = FRAMES_DIR / "img" / d.name / "votes.json"  # avec les images (frames.votes_path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"frames": {}, "cost_usd": 0.0}


def _frames(sid: str) -> list[Path]:
    """Images de la vidéo, sans celles qu'ffmpeg ajoute après sa fin (hors dataset, comme frames_of)."""
    duration = load_meta(SAMPLES_DIR / sid)["video"]["duration"]
    return [f for i, f in enumerate(sorted((FRAMES_DIR / "img" / sid).glob("*.jpg"))) if not duration or i < duration]


def videos() -> list[dict]:
    out = []
    for d in sorted(SAMPLES_DIR.glob("yt_*")):
        if not (d / "meta.json").exists():
            continue
        m, v = load_meta(d), _votes(d)["frames"]
        labels = [e["label"] for e in v.values()]
        out.append({"id": d.name, "title": m.get("title"), "channel": m.get("channel"), "url": m.get("url"),
                    "query": m.get("query"), "width": m["video"]["width"], "height": m["video"]["height"],
                    "duration": m["video"]["duration"], "images": len(_frames(d.name)),
                    "ajout": labels.count(1), "brute": labels.count(0), "sans": labels.count(None)})
    return out


def video(sid: str) -> dict:
    d = SAMPLES_DIR / Path(sid).name
    m, v = load_meta(d), _votes(d)
    frames = []
    for i, f in enumerate(_frames(d.name)):
        e = v["frames"].get(f.name)
        frames.append({"i": i, "t": i, "url": f"/img/{d.name}/{f.name}",
                       "label": e["label"] if e else None, "annotee": e is not None,
                       "votes": e["votes"] if e else {}})
    return {"id": d.name, "meta": m, "cost_usd": v["cost_usd"], "annotators": list(ANNOTATORS), "frames": frames}


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

    def _json(self, data) -> None:
        self._send(HTTPStatus.OK, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        if dev_reload.enabled():
            if u.path == "/__dev":
                return self._json(dev_reload.status(HERE / "index.html"))
            if u.path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, dev_reload.page(HERE / "index.html"), "text/html; charset=utf-8")
        if u.path == "/api/videos":
            return self._json(videos())
        if u.path == "/api/video":
            sid = parse_qs(u.query).get("id", [""])[0]
            if not (SAMPLES_DIR / Path(sid).name / "meta.json").exists():
                return self.send_error(HTTPStatus.NOT_FOUND)
            return self._json(video(sid))
        if u.path.startswith("/img/"):
            parts = unquote(u.path[5:]).split("/")
            if len(parts) != 2:
                return self.send_error(HTTPStatus.NOT_FOUND)
            f = FRAMES_DIR / "img" / Path(parts[0]).name / Path(parts[1]).name
            if not f.exists():
                return self.send_error(HTTPStatus.NOT_FOUND)
            return self._send(HTTPStatus.OK, f.read_bytes(), "image/jpeg")
        if u.path.endswith(".py"):
            return self.send_error(HTTPStatus.NOT_FOUND)
        return super().do_GET()


def main() -> None:
    p = argparse.ArgumentParser(prog="interface/dataset_viewer/server.py")
    p.add_argument("--port", type=int, default=8800)
    a = p.parse_args()
    print(f"visionneuse du dataset — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
