"""Page d'accueil des outils du projet : liens vers l'interface du modèle, le banc des annotateurs et
l'arbre des agents (utils/agents_tree.html, servi ici : c'est une page statique).

    .venv/Scripts/python.exe utils/home/server.py [--port 8770]
"""

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from utils import dev_reload  # noqa: E402

PAGES = {"/": HERE / "index.html", "/index.html": HERE / "index.html",
         "/agents_tree.html": ROOT / "utils" / "agents_tree.html"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/__dev" and dev_reload.enabled():
            # une seule date pour les deux pages : chacune se recharge quand l'une d'elles change
            newest = max(PAGES.values(), key=lambda p: p.stat().st_mtime if p.exists() else 0)
            return self._send(json.dumps(dev_reload.status(newest)).encode(), "application/json")
        page = PAGES.get(u.path)
        if page is None or not page.exists():
            return self.send_error(HTTPStatus.NOT_FOUND)
        body = dev_reload.page(page) if dev_reload.enabled() else page.read_bytes()
        self._send(body, "text/html; charset=utf-8")


def main() -> None:
    p = argparse.ArgumentParser(prog="utils/home/server.py")
    p.add_argument("--port", type=int, default=8770)
    a = p.parse_args()
    print(f"accueil — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
