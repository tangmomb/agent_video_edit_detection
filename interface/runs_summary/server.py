"""Résumé des runs : une ligne par run de runs/ (état, coût, durée, expériences, tests finaux, modèle
mis en production), et le détail court d'un run (étapes par agent avec leur durée et leur coût,
expériences entraînées). Complète les conversations des runs (interface/run_chat), qui relisent un
run message par message. Les journaux sont relus à chaque requête : un run en cours se suit en
direct. Aucun appel payant.

    .venv/Scripts/python.exe interface/runs_summary/server.py [--port 8830]
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from interface import dev_reload  # noqa: E402

RUNS = ROOT / "runs"
PROD = ROOT / "model_results" / "production.json"
PAGE = HERE / "index.html"
# Comme run_chat : un run « running » dont le journal n'a pas bougé depuis ce délai a été arrêté
# sans être clôturé, il est affiché « interrompu ».
STALE_S = 15 * 60
HEARTBEAT = ROOT / "data" / "batches.json"
_PROD_RE = re.compile(r"`(exp_\d+)` sur `(split_\d+)`")


def _stale(path: Path) -> bool:
    """Journal immobile depuis STALE_S. Pendant un batch d'annotation (jusqu'à 24 h), le dernier run
    reste vivant sans écrire : son signal de vie est la date de data/batches.json (batch.heartbeat)."""
    seen = path.stat().st_mtime
    if HEARTBEAT.exists() and path.parent.name == max(p.parent.name for p in RUNS.glob("*/pipeline.json")):
        seen = max(seen, HEARTBEAT.stat().st_mtime)
    return time.time() - seen > STALE_S


def _seconds(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()


def _production(run_dir: Path) -> dict | None:
    """Modèle mis en production en fin de run, lu dans la section du rapport écrite par le code."""
    report = run_dir / "report.md"
    if not report.exists():
        return None
    text = report.read_text(encoding="utf-8")
    part = text.split("## Modèle de production", 1)
    m = _PROD_RE.search(part[1]) if len(part) == 2 else None
    if not m:
        return None
    head = part[1][:400]
    kind = "conservé" if "conservé" in head else "non vérifié" if "NON VÉRIFIÉ" in head else "vérifié"
    return {"experiment_id": m[1], "split_id": m[2], "statut": kind}


def summarize(run_dir: Path) -> dict:
    path = run_dir / "pipeline.json"
    log = json.loads(path.read_text(encoding="utf-8"))
    r, steps = log.get("run", {}), log.get("steps", [])
    status = r.get("status")
    if status == "running" and _stale(path):
        status = "interrompu"
    last = steps[-1]["time"] if steps else r.get("started")
    now = datetime.now().isoformat(timespec="seconds")
    ended = r.get("ended") or (now if status == "running" else last)

    # Étapes : un appel d'agent par « start » ; le coût des « end » est cumulé sur la session.
    calls, open_calls = {}, {}
    for s in steps:
        if s["type"] == "start":
            calls[s["call"]] = {"call": s["call"], "agent": s["agent"], "parent": s.get("parent"),
                                "start": s["time"], "end": None, "task": (s.get("task") or "")[:160],
                                "tools": 0, "cost_cum": None}
            open_calls[s["call"]] = calls[s["call"]]
        elif s["type"] == "tool" and s.get("call") in calls:
            calls[s["call"]]["tools"] += 1
        elif s["type"] == "end" and s.get("call") in calls:
            c = calls[s["call"]]
            c["end"], c["cost_cum"] = s["time"], s.get("cost_usd")
            open_calls.pop(s["call"], None)
    # Coût d'une étape : par les tours (turn_cost_usd) de cet appel.
    turn_cost: dict[int, float] = {}
    for s in steps:
        if s["type"] == "turn" and s.get("turn_cost_usd") is not None:
            turn_cost[s["call"]] = turn_cost.get(s["call"], 0.0) + s["turn_cost_usd"]
    etapes = []
    for c in calls.values():
        if c["agent"] == "orchestrator":
            continue
        etapes.append({"agent": c["agent"], "task": c["task"], "start": c["start"],
                       "duration_s": _seconds(c["start"], c["end"] or ended), "en_cours": c["end"] is None,
                       "cost_usd": round(turn_cost.get(c["call"], 0.0), 4), "tools": c["tools"]})
    orch = next((c for c in calls.values() if c["agent"] == "orchestrator"), None)

    # Expériences, splits, collecte et tests finaux, lus dans les résultats d'outils.
    exps, splits, finals = [], [], []
    videos, annotation = 0, 0.0
    for s in steps:
        if s["type"] != "tool":
            continue
        res = s.get("result")
        try:
            inp = json.loads(s.get("input") or "{}")
        except (TypeError, ValueError):
            inp = {}
        if s["tool"] == "train_model" and isinstance(res, dict) and "experiment_id" in res:
            v = res.get("val", {})
            exps.append({"id": res["experiment_id"], "embeddings": inp.get("embeddings"), "model": inp.get("model", "logreg"),
                         "params": inp.get("params") or {}, "weights": inp.get("weights") or {},
                         "balance": inp.get("balance") or {}, "ap_par_chaine": v.get("ap_par_chaine", v.get("ap")),
                         "ap_globale": "ap_par_chaine" not in v,  # anciens runs : AP globale seulement
                         "recall": v.get("recall"), "fp": v.get("fp"), "fn": v.get("fn")})
        elif s["tool"] == "make_split" and isinstance(res, dict) and "split_id" in res:
            splits.append(res["split_id"])
        elif s["tool"] == "generate_data" and isinstance(res, dict):
            videos += res.get("videos_collectees") or 0
            annotation += res.get("cout_annotation_usd") or 0.0
        elif s["tool"] == "final_test" and isinstance(res, dict) and "test" in res:
            t, target = res["test"], res.get("precision_visee")
            # Les anciens runs n'avaient pas de verdict : même règle que lab._test_status.
            statut = res.get("statut") or (None if target is None or t.get("precision") is None else
                                           "réussi" if t["precision"] >= target else "échoué")
            finals.append({"id": res.get("experiment_id"), "statut": statut,
                           "precision_visee": res.get("precision_visee"),
                           "precision": t.get("precision"), "recall": t.get("recall")})
    best = max((e for e in exps if e["ap_par_chaine"] is not None), key=lambda e: e["ap_par_chaine"], default=None)
    current = next(iter(open_calls.values()), None) if status == "running" else None
    return {
        "id": run_dir.name, "n": run_dir.name.split("_")[0], "status": status, "started": r.get("started"),
        "ended": r.get("ended"), "duration_s": _seconds(r.get("started"), ended),
        "cost_usd": r.get("cost_usd"), "budget_usd": r.get("budget_usd"), "annotation_usd": round(annotation, 4),
        "agent_model": r.get("agent_model"), "tokens": r.get("tokens"), "videos_collectees": videos,
        "splits": splits, "experiences": exps, "best": best, "final_tests": finals,
        "production": _production(run_dir), "etapes": etapes,
        "orchestrateur_s": _seconds(orch["start"], orch["end"] or ended) if orch else None,
        "en_cours": {"agent": current["agent"], "depuis": current["start"]} if current else None,
        "dernier_evenement": last, "error": r.get("error"),
        "conclusion": " ".join((r.get("report") or "").split())[:300],
    }


def runs() -> dict:
    out, unreadable = [], []
    for path in sorted(RUNS.glob("*/pipeline.json"), reverse=True):
        try:
            out.append(summarize(path.parent))
        except (OSError, ValueError, KeyError):
            unreadable.append(path.parent.name)  # journal vide, abîmé ou en cours d'écriture
    prod = json.loads(PROD.read_text(encoding="utf-8")) if PROD.exists() else None
    return {"runs": out, "illisibles": unreadable, "production": prod}


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
        return self.send_error(HTTPStatus.NOT_FOUND)


def main() -> None:
    p = argparse.ArgumentParser(prog="interface/runs_summary/server.py")
    p.add_argument("--port", type=int, default=8830)
    a = p.parse_args()
    print(f"résumé des runs — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
