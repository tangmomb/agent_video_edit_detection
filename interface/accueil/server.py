"""Panneau de contrôle du projet : une seule adresse, un menu (burger, en bas à gauche) pour passer
d'un outil à l'autre. Chaque outil garde son dossier dans interface/ et, s'il a un serveur, son port :
le panneau l'affiche dans un cadre (page /#/<id>). Les pages statiques (arbre des agents, choix de la
production) sont servies ici.

    .venv/Scripts/python.exe interface/accueil/server.py [--port 8770]
"""

import argparse
import json
import os
import socket
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
# Les outils du menu, dans l'ordre (le premier s'ouvre à l'arrivée) : id de la page (/#/<id>), nom, description, adresse affichée dans
# le cadre et, pour un outil à serveur, le port dont on vérifie qu'il répond (voyant du menu).
TOOLS = [
    {"id": "runs", "name": "Résumé des runs", "icon": "📋", "url": "http://localhost:8830/",
     "port": 8830,
     "desc": "Une ligne par run : run en cours, coût, durée, temps et coût de chaque étape, expériences, "
             "tests finaux et modèle mis en production."},
    {"id": "modele", "name": "Modèle en production", "icon": "▶", "url": "http://localhost:8780/",
     "port": 8780,
     "desc": "Envoie une vidéo et vois les images où le classifieur détecte un ajout de montage, avec un "
             "seuil réglable. Le modèle met quelques secondes à se charger."},
    {"id": "conversations", "name": "Conversations des runs", "icon": "💬", "url": "http://localhost:8820/",
     "port": 8820,
     "desc": "Relis un run comme une discussion entre les agents ; clique une bulle pour voir l'appel complet "
             "au modèle. Un run en cours se suit en direct."},
    {"id": "agents", "name": "Agents et outils", "icon": "🌳", "url": "/agents_tree.html",
     "desc": "L'arbre des agents, ce que reçoit chaque modèle et des exemples d'appels réels. Régénéré par "
             "interface/agents_tree/agents_tree.py."},
    {"id": "production", "name": "Le parcours jusqu'au modèle de production", "icon": "🏁",
     "url": "/choix_production.html",
     "desc": "Du split créé par les agents au modèle que le code met en production à la fin d'un run."},
    {"id": "dataset", "name": "Visionneuse du dataset", "icon": "🎞", "url": "http://localhost:8800/",
     "port": 8800,
     "desc": "Parcours les vidéos du dataset : étiquette et votes des trois annotateurs pour chaque image."},
    {"id": "banc", "name": "Banc des annotateurs", "icon": "⚖", "url": "http://localhost:8790/",
     "port": 8790,
     "desc": "Compare les réponses, le coût et la durée des LLM annotateurs sur une même vidéo (appels payants)."},
    {"id": "saut", "name": "Test du saut d'images", "icon": "⏭", "url": "http://localhost:8810/",
     "port": 8810,
     "desc": "Les images que l'annotation enverrait aux annotateurs et celles sautées comme quasi identiques, "
             "avec le seuil réglable."},
]
# Sur le VPS (INTERFACE_PROXIED=1), Caddy est le seul point d'entrée : chaque outil est servi sous
# /<id>/ (prive : derrière un mot de passe, aucun aujourd'hui), et local_seulement : non publié (aucun aujourd'hui).
PROXIED = bool(os.environ.get("INTERFACE_PROXIED"))
PUBLISHED = [{**t, "url": f"/{t['id']}/" if PROXIED and t.get("port") else t["url"],
              "prive": PROXIED and t.get("prive", False)}
             for t in TOOLS if not (PROXIED and t.get("local_seulement"))]
PAGES = {"/": HERE / "index.html", "/index.html": HERE / "index.html",
         "/agents_tree.html": INTERFACE / "agents_tree" / "agents_tree.html",
         "/choix_production.html": INTERFACE / "choix_production" / "index.html"}


def _listening(port: int) -> bool:
    """Un outil répond-il ? Vérifié ici et non par le navigateur : sonder une page sous mot de passe
    ferait apparaître la demande de mot de passe dès l'arrivée sur le panneau."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


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
            return self._send(json.dumps(PUBLISHED, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        if u.path == "/api/status":
            status = {t["id"]: _listening(t["port"]) for t in PUBLISHED if t.get("port")}
            return self._send(json.dumps(status).encode(), "application/json")
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
