"""Conversations des runs : chaque run de runs/ relu comme une discussion de groupe entre les agents
(orchestrateur, sous-agents) et les outils du labo, à partir de son pipeline.json. Les journaux sont
relus à chaque requête : un run en cours se suit en direct. Aucun appel payant.

    .venv/Scripts/python.exe interface/run_chat/server.py [--port 8820]
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
from interface import dev_reload  # noqa: E402

RUNS = ROOT / "runs"
PAGE = HERE / "index.html"
MAX_RESULT = 6000  # caractères envoyés d'un résultat d'outil (la page coupe encore à l'affichage)
# Un run « running » dont le journal n'a pas bougé depuis ce délai a été arrêté sans être clôturé
# (run 006, interrompu à la main) : affiché « interrompu » plutôt qu'en cours pour toujours.
STALE_S = 15 * 60
OLD_SIGNS = (ROOT / "data" / "batches.json", ROOT / "data" / "videos_samples")


def _stale(path: Path) -> bool:
    """Aucun signe de vie depuis STALE_S : journal immobile et fichier alive (Session._alive) pas touché.
    Les runs d'avant ce fichier : batches.json et l'arrivée des vidéos, pour le dernier run seulement."""
    alive = path.parent / "alive"
    seen = [path, alive] if alive.exists() else [path]
    if path.parent.name == max(p.parent.name for p in RUNS.glob("*/pipeline.json")):
        seen += [f for f in OLD_SIGNS if f.exists()]
    return time.time() - max(f.stat().st_mtime for f in seen) > STALE_S


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def _cut(text: str) -> str:
    return text if len(text) <= MAX_RESULT else text[:MAX_RESULT] + f"\n… ({len(text) - MAX_RESULT} caractères de plus)"


def _items(content: list, run: str) -> list[dict]:
    """Texte et images (« <image chemin> » dans le journal) d'un résultat d'outil ou d'un message du code."""
    out = []
    for c in map(str, content):
        if c.startswith("<image ") and c.endswith(">"):
            out.append({"img": f"img/{run}/{Path(c[7:-1]).name}"})
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
                out.append({**base, "k": "turn", "text": text, "thinking": thinking, "calls": calls, "turn_n": s["n"]})
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
            final, thinking, turn_n = s.get("final") or "", "", None
            last = out[-1] if out else None
            # Le rapport est le texte du dernier tour (sans outil), suivi de la vérification
            # automatique : ce tour devient la bulle du rapport, pour qu'une bulle reste un appel au
            # modèle au lieu d'afficher deux fois le même texte.
            if (last and last["k"] == "turn" and last["agent"] == s["agent"] and not last["calls"]
                    and final.startswith(last["text"].strip()[:200])):
                popped = out.pop()
                thinking, turn_n = popped["thinking"], popped["turn_n"]
            out.append({**base, "k": "end", "text": final, "thinking": thinking, "turn_n": turn_n,
                        "cost": s.get("cost_usd"),
                        "dur": s.get("duration_s"), "tools": s.get("tools_called")})
    return out


def _tool_content(result, run: str) -> list[dict]:
    """Message d'outil tel que le runner l'a renvoyé au modèle : le JSON du résultat, ou, pour un
    résultat avec images, son texte suivi d'une note (les images et leurs légendes partent dans le
    message user suivant)."""
    if not isinstance(result, list):
        return [{"text": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)}]
    items = [str(c) for c in result]
    imgs = [k for k, c in enumerate(items) if c.startswith("<image ")]
    captions = {k - 1 for k in imgs if k and not items[k - 1].startswith("<image ")}
    text = "\n".join(c for k, c in enumerate(items) if k not in captions and k not in imgs)
    if imgs:
        text = (text + "\n" if text else "") + f"[{len(imgs)} image(s), chacune précédée de sa légende, dans le message suivant]"
    return [{"text": text}]


def call_detail(log: dict, run: str, n: int) -> dict:
    """Tout l'appel au modèle d'une bulle (le tour n du journal), reconstitué comme le runner l'a
    envoyé : prompt système, schémas d'outils, messages depuis la consigne (tours précédents de
    l'agent, résultats d'outils, messages du code), puis la réponse, ses tokens et son coût."""
    steps = log["steps"]
    turn = next(s for s in steps if s["n"] == n and s["type"] == "turn")
    mine = [s for s in steps if s["call"] == turn["call"] and s["n"] < n]
    agent = log.get("agents", {}).get(turn["agent"], {})
    start = next((s for s in mine if s["type"] == "start"), {})
    messages = [{"role": "user", "items": [{"text": start.get("task", "")}]}]
    for s in mine:
        if s["type"] == "turn":
            messages.append({"role": "assistant", "thinking": "\n\n".join(s.get("thinking") or []),
                             "items": [{"text": "\n\n".join(s.get("text") or [])}],
                             "calls": s.get("tool_calls") or []})
        elif s["type"] == "tool":
            messages.append({"role": "tool", "name": s["tool"], "items": _tool_content(s.get("result"), run)})
        elif s["type"] == "user":
            messages.append({"role": "user", "items": _items(s.get("content") or [], run)})
    return {"agent": turn["agent"], "turn": turn.get("turn"), "model": start.get("model"),
            "reasoning": start.get("reasoning"), "max_turns": start.get("max_turns"),
            "system": agent.get("system", ""), "tools": agent.get("tools", []), "messages": messages,
            "response": {"thinking": "\n\n".join(turn.get("thinking") or []),
                         "text": "\n\n".join(turn.get("text") or []), "calls": turn.get("tool_calls") or [],
                         "stop": turn.get("stop")},
            "tokens": turn.get("tokens"), "cost": turn.get("turn_cost_usd"), "dur": turn.get("duration_s")}


def _reply(raw: dict) -> dict:
    """Message de l'assistant tel que le runner le rejoue au tour suivant (sans les champs vides ni
    l'annotation interne de LiteLLM)."""
    msg = dict(((raw or {}).get("choices") or [{}])[0].get("message") or {})
    msg.pop("provider_specific_fields", None)
    return {k: v for k, v in msg.items() if v is not None}


def raw_call(log: dict, n: int) -> dict:
    """Requête et réponse brutes de l'appel du tour n. Exactes pour les runs qui les journalisent
    (request et raw_response à chaque tour) ; sinon reconstituées depuis le contenu du journal, au
    format de l'API (sans les id des appels d'outils, jamais enregistrés)."""
    steps = log["steps"]
    turn = next(s for s in steps if s["n"] == n and s["type"] == "turn")
    mine = [s for s in steps if s["call"] == turn["call"] and s["n"] <= n]
    tools = log.get("agents", {}).get(turn["agent"], {}).get("tools", [])
    if "request" in turn:
        messages = []
        for s in mine:
            if s["type"] == "turn":
                messages += s["request"]["new_messages"]
                if s["n"] < n:
                    messages.append(_reply(s.get("raw_response")))
        return {"exact": True, "request": {**turn["request"]["params"], "messages": messages, "tools": tools},
                "response": turn.get("raw_response")}
    start = next((s for s in mine if s["type"] == "start"), {})
    messages = [{"role": "system", "content": log.get("agents", {}).get(turn["agent"], {}).get("system", "")},
                {"role": "user", "content": start.get("task", "")}]
    for s in mine:
        if s["type"] == "turn" and s["n"] < n:
            msg = {"role": "assistant", "content": "\n\n".join(s.get("text") or [])}
            if s.get("thinking"):
                msg["reasoning_content"] = "\n\n".join(s["thinking"])
            if s.get("tool_calls"):
                msg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["input"]}}
                                     for c in s["tool_calls"]]
            messages.append(msg)
        elif s["type"] == "tool":
            messages.append({"role": "tool", "name": s["tool"],
                             "content": _tool_content(s.get("result"), "")[0]["text"]})
        elif s["type"] == "user":
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": c}} if str(c).startswith("<image ") else {"type": "text", "text": c}
                for c in s.get("content") or []]})
    msg = {"role": "assistant", "content": "\n\n".join(turn.get("text") or [])}
    if turn.get("thinking"):
        msg["reasoning_content"] = "\n\n".join(turn["thinking"])
    if turn.get("tool_calls"):
        msg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["input"]}}
                             for c in turn["tool_calls"]]
    t = turn.get("tokens") or {}
    return {"exact": False,
            "request": {"model": start.get("model"), "max_tokens": 16000, "messages": messages, "tools": tools},
            "response": {"choices": [{"finish_reason": turn.get("stop"), "message": msg}],
                         "usage": {"prompt_tokens": t.get("input"), "completion_tokens": t.get("output"),
                                   "prompt_tokens_details": {"cached_tokens": t.get("cached_input")}}}}


def _log(run: str) -> dict:
    path = RUNS / Path(run).name / "pipeline.json"
    log = json.loads(path.read_text(encoding="utf-8"))
    r = log.setdefault("run", {})
    if r.get("status") == "running" and _stale(path):
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
        if u.path == "/api/call":
            q = parse_qs(u.query)
            try:
                run = q.get("id", [""])[0]
                return self._json(call_detail(_log(run), run, int(q.get("n", ["0"])[0])))
            except (FileNotFoundError, StopIteration, ValueError):
                return self.send_error(HTTPStatus.NOT_FOUND)
        if u.path == "/api/raw":
            q = parse_qs(u.query)
            try:
                return self._json(raw_call(_log(q.get("id", [""])[0]), int(q.get("n", ["0"])[0])))
            except (FileNotFoundError, StopIteration, ValueError):
                return self.send_error(HTTPStatus.NOT_FOUND)
        if u.path.startswith("/img/"):
            parts = unquote(u.path[5:]).split("/")
            f = RUNS / Path(parts[0]).name / "images" / Path(parts[-1]).name if len(parts) == 2 else None
            if f is None or not f.exists():
                return self.send_error(HTTPStatus.NOT_FOUND)
            return self._send(f.read_bytes(), "image/jpeg")
        return self.send_error(HTTPStatus.NOT_FOUND)


def main() -> None:
    p = argparse.ArgumentParser(prog="interface/run_chat/server.py")
    p.add_argument("--port", type=int, default=8820)
    a = p.parse_args()
    print(f"conversations des runs — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
