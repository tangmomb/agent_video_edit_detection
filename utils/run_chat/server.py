"""Conversations des runs : chaque run de runs/ relu comme une discussion de groupe entre les agents
(orchestrateur, sous-agents) et les outils du labo, à partir de son pipeline.json. Les journaux sont
relus à chaque requête : un run en cours se suit en direct. Aucun appel payant.

    .venv/Scripts/python.exe utils/run_chat/server.py [--port 8820]
"""

import argparse
import json
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from utils import dev_reload  # noqa: E402

RUNS = ROOT / "runs"
PAGE = HERE / "index.html"
MAX_RESULT = 6000  # caractères envoyés d'un résultat d'outil (la page coupe encore à l'affichage)
# Un run « running » dont le journal n'a pas bougé depuis ce délai a été arrêté sans être clôturé
# (run 006, interrompu à la main) : affiché « interrompu » plutôt qu'en cours pour toujours.
STALE_S = 15 * 60


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def _cut(text: str) -> str:
    return text if len(text) <= MAX_RESULT else text[:MAX_RESULT] + f"\n… ({len(text) - MAX_RESULT} caractères de plus)"


def _items(content: list, run: str) -> list[dict]:
    """Texte et images (« <image chemin> » dans le journal) d'un résultat d'outil ou d'un message du code."""
    out = []
    for c in map(str, content):
        if c.startswith("<image ") and c.endswith(">"):
            out.append({"img": f"/img/{run}/{Path(c[7:-1]).name}"})
        else:
            out.append({"text": _cut(c)})
    return out


def messages(log: dict, run: str) -> list[dict]:
    """Les étapes du journal, dans l'ordre, en messages : consigne reçue (task), tour d'un agent
    (turn : texte, réflexion, appels d'outils), résultat d'outil (tool), message ajouté par le code
    (code) et rapport final d'un agent (end)."""
    agent_of = {}  # appel d'agent -> nom de l'agent
    out, shown = [], set()
    for s in log["steps"]:
        base = {"n": s["n"], "time": s["time"][11:19], "agent": s["agent"]}
        kind = s["type"]
        if kind == "start":
            agent_of[s["call"]] = s["agent"]
            out.append({**base, "k": "task", "from": agent_of.get(s["parent"], "toi"), "text": s["task"]})
        elif kind == "turn":
            text, thinking = "\n\n".join(s.get("text") or []), "\n\n".join(s.get("thinking") or [])
            calls = [{"name": c["name"], "input": c["input"]} for c in s.get("tool_calls") or []]
            if text or thinking or calls:
                out.append({**base, "k": "turn", "text": text, "thinking": thinking, "calls": calls})
        elif kind == "tool":
            if s["tool"].startswith("ask_") and not s["error"]:
                continue  # le rapport du sous-agent arrive déjà avec son « end »
            result = s.get("result")
            items = _items(result, run) if isinstance(result, list) else [{"text": _cut(_text(result))}]
            shown.update(i["img"] for i in items if "img" in i)
            if s.get("traceback"):
                items.append({"text": _cut(s["traceback"])})
            out.append({**base, "k": "tool", "tool": s["tool"], "input": s.get("input") or "",
                        "items": items, "error": bool(s["error"]), "dur": s.get("duration_s")})
        elif kind == "user":
            items = _items(s.get("content") or [], run)
            imgs = [i["img"] for i in items if "img" in i]
            if imgs and all(i in shown for i in imgs):
                continue  # images d'un show_examples, déjà montrées avec son résultat
            out.append({**base, "k": "code", "items": items})
        elif kind == "end":
            out.append({**base, "k": "end", "text": s.get("final") or "", "cost": s.get("cost_usd"),
                        "dur": s.get("duration_s"), "tools": s.get("tools_called")})
    return out


def _log(run: str) -> dict:
    path = RUNS / Path(run).name / "pipeline.json"
    log = json.loads(path.read_text(encoding="utf-8"))
    r = log.setdefault("run", {})
    if r.get("status") == "running" and time.time() - path.stat().st_mtime > STALE_S:
        r["status"] = "interrompu"
    return log


def runs() -> list[dict]:
    """Les runs, du plus récent au plus ancien, avec leur état et leur coût."""
    out = []
    for path in sorted(RUNS.glob("*/pipeline.json"), reverse=True):
        try:
            r = _log(path.parent.name)["run"]
        except (OSError, ValueError):
            continue  # journal en cours d'écriture : il reviendra à la requête suivante
        report = " ".join((r.get("report") or r.get("error") or "").split())
        out.append({"id": path.parent.name, "status": r.get("status"), "started": r.get("started"),
                    "cost": r.get("cost_usd"), "budget": r.get("budget_usd"), "model": r.get("agent_model"),
                    "calls": r.get("agent_calls"), "preview": report[:140]})
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status=HTTPStatus.OK) -> None:
        self._send(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/__dev" and dev_reload.enabled():
            return self._json(dev_reload.status(PAGE))
        if u.path in ("/", "/index.html"):
            body = dev_reload.page(PAGE) if dev_reload.enabled() else PAGE.read_bytes()
            return self._send(body, "text/html; charset=utf-8")
        if u.path == "/api/runs":
            return self._json(runs())
        if u.path == "/api/run":
            run = parse_qs(u.query).get("id", [""])[0]
            try:
                log = _log(run)
            except FileNotFoundError:
                return self.send_error(HTTPStatus.NOT_FOUND)
            except ValueError:  # écriture en cours : la page redemandera
                return self._json({"error": "journal en cours d'écriture"}, HTTPStatus.SERVICE_UNAVAILABLE)
            return self._json({**log.get("run", {}), "id": run, "msgs": messages(log, run)})
        if u.path.startswith("/img/"):
            parts = unquote(u.path[5:]).split("/")
            f = RUNS / Path(parts[0]).name / "images" / Path(parts[-1]).name if len(parts) == 2 else None
            if f is None or not f.exists():
                return self.send_error(HTTPStatus.NOT_FOUND)
            return self._send(f.read_bytes(), "image/jpeg")
        return self.send_error(HTTPStatus.NOT_FOUND)


def main() -> None:
    p = argparse.ArgumentParser(prog="utils/run_chat/server.py")
    p.add_argument("--port", type=int, default=8820)
    a = p.parse_args()
    print(f"conversations des runs — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
