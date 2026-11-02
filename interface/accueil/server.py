"""Panneau de contrôle du projet : une seule adresse, un menu (burger, en bas à gauche) pour passer
d'un outil à l'autre. Chaque outil garde son dossier dans interface/ et, s'il a un serveur, son port :
le panneau l'affiche dans un cadre (page /#/<id>). Les pages statiques (arbre des agents, choix de la
production) sont servies ici.

    .venv/Scripts/python.exe interface/accueil/server.py [--port 8770]
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
from interface import dev_reload  # noqa: E402

INTERFACE = HERE.parent
# Les outils du menu, dans l'ordre : id de la page (/#/<id>), nom, description, adresse affichée dans
# le cadre et, pour un outil à serveur, l'adresse dont on vérifie qu'elle répond (voyant du menu).
TOOLS = [
    {"id": "accueil", "name": "Accueil", "icon": "⌂", "url": "/outils.html",
     "desc": "Tous les outils et leur état."},
    {"id": "modele", "name": "Interface du modèle", "icon": "▶", "url": "http://localhost:8780/",
     "check": "http://localhost:8780/",
     "desc": "Envoie une vidéo et vois les images où le classifieur détecte un ajout de montage ; annotation "
             "par les trois LLM et correction à la main. Le modèle met quelques secondes à se charger."},
    {"id": "conversations", "name": "Conversations des runs", "icon": "💬", "url": "http://localhost:8820/",
     "check": "http://localhost:8820/",
     "desc": "Relis un run comme une discussion entre les agents ; clique une bulle pour voir l'appel complet "
             "au modèle. Un run en cours se suit en direct."},
    {"id": "agents", "name": "Agents et outils", "icon": "🌳", "url": "/agents_tree.html",
     "desc": "L'arbre des agents, ce que reçoit chaque modèle et des exemples d'appels réels. Régénéré par "
             "interface/agents_tree/agents_tree.py."},
    {"id": "production", "name": "Choix de la production", "icon": "🏁", "url": "/choix_production.html",
     "desc": "Comment le code choisit le modèle mis en production à la fin d'un run."},
    {"id": "dataset", "name": "Visionneuse du dataset", "icon": "🎞", "url": "http://localhost:8800/",
     "check": "http://localhost:8800/",
     "desc": "Parcours les vidéos du dataset : étiquette et votes des trois annotateurs pour chaque image."},
    {"id": "banc", "name": "Banc des annotateurs", "icon": "⚖", "url": "http://localhost:8790/",
     "check": "http://localhost:8790/",
     "desc": "Compare les réponses, le coût et la durée des LLM annotateurs sur une même vidéo (appels payants)."},
    {"id": "saut", "name": "Test du saut d'images", "icon": "⏭", "url": "http://localhost:8810/",
     "check": "http://localhost:8810/",
     "desc": "Les images que l'annotation enverrait aux annotateurs et celles sautées comme quasi identiques, "
             "avec le seuil réglable."},
]
PAGES = {"/": HERE / "index.html", "/index.html": HERE / "index.html", "/outils.html": HERE / "outils.html",
         "/agents_tree.html": INTERFACE / "agents_tree" / "agents_tree.html",
         "/choix_production.html": INTERFACE / "choix_production" / "index.html"}


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
            # une seule date pour toutes les pages : chacune se recharge quand l'une d'elles change
            newest = max(PAGES.values(), key=lambda p: p.stat().st_mtime if p.exists() else 0)
            return self._send(json.dumps(dev_reload.status(newest)).encode(), "application/json")
        if u.path == "/api/tools":
            return self._send(json.dumps(TOOLS, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        page = PAGES.get(u.path)
        if page is None or not page.exists():
            return self.send_error(HTTPStatus.NOT_FOUND)
        body = dev_reload.page(page) if dev_reload.enabled() else page.read_bytes()
        self._send(body, "text/html; charset=utf-8")


def main() -> None:
    p = argparse.ArgumentParser(prog="interface/accueil/server.py")
    p.add_argument("--port", type=int, default=8770)
    a = p.parse_args()
    print(f"panneau de contrôle — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
