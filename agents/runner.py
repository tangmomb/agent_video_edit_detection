"""Boucle tool use commune aux quatre agents.

Un `Agent` = un prompt système + une liste d'outils + un modèle. Une `Session` exécute des agents
sur l'API Mistral (chat completions + function calling) avec un budget partagé et journalise tout
le pipeline dans runs/<horodatage>/ (à la racine du projet) : pipeline.json (réécrit à chaque
étape), plus report.md pour le rapport final.

pipeline.json = {"run": {...paramètres, statut, coût, tokens...}, "steps": [...]}, les étapes
dans l'ordre chronologique. Chaque étape porte `call` (n° d'appel d'agent) et `parent` (appel
qui l'a délégué), d'où l'arbre orchestrateur → sous-agents. Types d'étape :
- start : un agent reçoit une tâche ;
- turn : une réponse du modèle (raisonnement, texte, outils demandés, tokens, coût cumulé) ;
- tool : un appel d'outil (paramètres, résultat complet, sortie console, durée, erreur) ;
- end : rapport final de l'agent ;
- error : exception inattendue (avec traceback).

Chaque appel d'agent repart d'une conversation vide : l'état durable (dataset, expériences,
production) vit sur disque, pas dans les conversations.
"""

import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from edit_generator.config import ROOT

RUNS_DIR = ROOT / "runs"

# Identifiants épinglés (pas d'alias -latest) pour qu'un changement de modèle côté Mistral ne
# modifie pas le comportement en silence.
LARGE = "mistral-large-4-0"   # Mistral Large 4 (v26.10) : raisonnement, vision, 1M de contexte
SMALL = "mistral-small-2603"  # Mistral Small 4 (v26.03) : raisonnement, vision, 256K
# $/million de tokens (docs.mistral.ai/models, octobre 2026). cached_input : tokens d'entrée
# servis depuis le cache, quand l'API les signale et que Mistral publie un prix.
PRICES = {
    LARGE: {"input": 1.36, "cached_input": 0.14, "output": 4.18},
    SMALL: {"input": 0.15, "output": 0.60},
}


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]  # renvoie str, liste de blocs (texte/images) ou objet JSON-sérialisable

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.input_schema}}

    def call(self, tool_input: dict) -> str | list:
        result = self.fn(**tool_input)
        if isinstance(result, str) or (isinstance(result, list) and result and isinstance(result[0], dict)
                                       and result[0].get("type") in ("text", "image")):
            return result
        return json.dumps(result, ensure_ascii=False)


@dataclass
class Agent:
    name: str
    system: str
    tools: list[Tool] = field(default_factory=list)
    max_turns: int = 25
    model: str = LARGE  # Small 4 inventait des actions et des sorties de commandes (run du 22/10/2026)
    reasoning: str = "high"  # reasoning_effort Mistral : "high" (trace de raisonnement) ou "none"


class _Capture(io.TextIOBase):
    """Recopie les print() d'un outil dans la console et les garde pour pipeline.json.

    Écrit directement dans la vraie console : un outil imbriqué (délégation) ne recopie donc pas
    la sortie de ses sous-outils, qui ont leur propre capture.
    """

    def __init__(self):
        self.buf = io.StringIO()

    def write(self, s: str) -> int:
        self.buf.write(s)
        _console(s)
        return len(s)


def _console(s: str) -> None:
    out = sys.__stdout__
    if out is None:
        return
    enc = out.encoding or "utf-8"  # la console Windows (cp1252) ne sait pas tout afficher
    out.write(s.encode(enc, "replace").decode(enc))
    out.flush()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Session:
    def __init__(self, budget_usd: float, client=None, run_dir: Path | None = None, verbose: bool = True,
                 params: dict | None = None):
        if client is None:
            from mistralai.client import Mistral
            client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.client = client
        self.budget_usd = budget_usd
        self.spent = 0.0
        self.calls = 0
        self.verbose = verbose
        self.run_dir = run_dir or RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._stack: list[int] = []  # appels d'agent en cours (le dernier est le plus profond)
        self.trace = {
            "run": {"status": "running", "started": _now(), "ended": None,
                    "budget_usd": budget_usd, **(params or {}), "cost_usd": 0.0,
                    "tokens": {"input": 0, "cached_input": 0, "output": 0},
                    "agent_calls": 0, "report": None, "error": None},
            "steps": [],
        }
        self._save()

    @property
    def budget_low(self) -> bool:
        return self.spent >= 0.8 * self.budget_usd

    @staticmethod
    def _cost(model: str, tokens: dict) -> float:
        p = PRICES[model]
        cached = tokens["cached_input"] if "cached_input" in p else 0
        return ((tokens["input"] - cached) * p["input"] + cached * p.get("cached_input", 0)
                + tokens["output"] * p["output"]) / 1e6

    def _say(self, msg: str) -> None:
        if self.verbose:
            _console(msg + "\n")

    def _save(self) -> None:
        run = self.trace["run"]
        run["cost_usd"] = round(self.spent, 4)
        run["agent_calls"] = self.calls
        path = self.run_dir / "pipeline.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)

    def _step(self, call: int, agent: str, type_: str, **data) -> None:
        parent = self._stack[-2] if len(self._stack) > 1 else None
        self.trace["steps"].append({"n": len(self.trace["steps"]) + 1, "time": _now(), "call": call,
                                    "parent": parent, "agent": agent, "type": type_, **data})
        self._save()

    def finish(self, report: str | None = None, error: BaseException | None = None) -> None:
        """Clôt pipeline.json : statut, heure de fin, rapport ou erreur."""
        run = self.trace["run"]
        run["ended"] = _now()
        run["report"] = report
        if error is not None:
            run["status"] = "error"
            run["error"] = f"{type(error).__name__}: {error}"
        else:
            run["status"] = "done"
        self._save()

    def run(self, agent: Agent, task: str) -> str:
        """Exécute `agent` sur `task` jusqu'à sa réponse finale ; renvoie le texte de cette réponse."""
        self.calls += 1
        call = self.calls
        self._stack.append(call)
        step = lambda type_, **data: self._step(call, agent.name, type_, **data)
        t_agent = time.perf_counter()

        step("start", task=task, max_turns=agent.max_turns, model=agent.model, reasoning=agent.reasoning)
        self._say(f"\n=== {agent.name} ===\n{task}")
        tools = {t.name: t for t in agent.tools}
        messages = [{"role": "system", "content": agent.system}, {"role": "user", "content": task}]
        final = ""
        used: list[tuple[str, str | None]] = []  # (outil, erreur ou None) pour la vérification
        try:
            for turn in range(1, agent.max_turns + 1):
                t0 = time.perf_counter()
                response = self.client.chat.complete(
                    model=agent.model,
                    max_tokens=16000,
                    messages=messages,
                    tools=[t.schema() for t in agent.tools],
                    tool_choice="auto",
                    reasoning_effort=agent.reasoning,
                )
                choice = response.choices[0]
                msg = choice.message
                tokens = _tokens(response.usage)
                cost = self._cost(agent.model, tokens)
                self.spent += cost
                for k, v in tokens.items():
                    self.trace["run"]["tokens"][k] += v
                messages.append(msg)  # message complet, raisonnement compris (à rejouer tel quel)
                thinking, texts = _split(msg.content)
                calls = msg.tool_calls or []
                for t in texts:
                    self._say(f"[{agent.name}] {t}")
                step("turn", turn=turn, stop=choice.finish_reason, thinking=thinking, text=texts,
                     tool_calls=[{"name": c.function.name, "input": c.function.arguments} for c in calls],
                     tokens=tokens, turn_cost_usd=round(cost, 5), cost_usd=round(self.spent, 4),
                     duration_s=round(time.perf_counter() - t0, 2))

                if not calls:
                    final = "\n\n".join(texts)
                    if choice.finish_reason == "length":
                        final += "\n\n[réponse tronquée : limite de tokens atteinte]"
                    break

                images = []  # l'API n'accepte les images que dans un message user
                for c in calls:
                    name, args = c.function.name, c.function.arguments
                    self._say(f"[{agent.name} → {name}] {str(args)[:200]}")
                    t0 = time.perf_counter()
                    capture = _Capture()
                    saved_stdout, sys.stdout = sys.stdout, capture
                    tb = None
                    try:
                        if name not in tools:
                            raise ValueError(f"outil inconnu : {name}")
                        tool_input = json.loads(args) if isinstance(args, str) else args
                        content = tools[name].call(tool_input or {})
                    except Exception as e:
                        content = f"Erreur : {type(e).__name__}: {e}"
                        tb = traceback.format_exc()
                    finally:
                        sys.stdout = saved_stdout
                    if isinstance(content, list):
                        imgs = [b for b in content if b["type"] == "image"]
                        images += [{"type": "image_url",
                                    "image_url": f"data:{b['source']['media_type']};base64,{b['source']['data']}"}
                                   for b in imgs]
                        text = "\n".join(b["text"] for b in content if b["type"] == "text")
                        if imgs:
                            text += f"\n[{len(imgs)} image(s) dans le message suivant]"
                    else:
                        text = content
                    messages.append({"role": "tool", "name": name, "content": text, "tool_call_id": c.id})
                    result = _loggable(content)
                    used.append((name, _failure(content, result, tb)))
                    step("tool", tool=name, input=args, result=result, error=tb is not None,
                         traceback=tb, stdout=capture.buf.getvalue() or None,
                         duration_s=round(time.perf_counter() - t0, 2))

                follow_up = list(images)
                remaining = agent.max_turns - turn
                if remaining <= 2 or self.budget_low:
                    follow_up.append({"type": "text", "text":
                                      f"[{remaining} tours restants, {self.spent:.2f}/{self.budget_usd:.2f} $ "
                                      "dépensés sur la session. Termine maintenant et rends ton rapport.]"})
                if follow_up:
                    messages.append({"role": "user", "content": follow_up})
                if self.spent >= self.budget_usd:
                    final = f"Agent arrêté : budget de session ({self.budget_usd} $) atteint."
                    break
            else:
                final = "Agent arrêté : nombre maximal de tours atteint sans rapport."
            if len(self._stack) > 1:  # sous-agent : son rapport repart vers l'orchestrateur
                final += "\n\n" + _verification(agent.name, used)
        except BaseException as e:
            step("error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            raise
        else:
            step("end", final=final, tools_called=len(used), cost_usd=round(self.spent, 4),
                 duration_s=round(time.perf_counter() - t_agent, 2))
        finally:
            self._stack.pop()
        return final


def _failure(content, result, tb) -> str | None:
    """Message d'erreur si l'appel d'outil a échoué (exception ou {"error": ...} renvoyé), sinon None."""
    if tb is not None:
        return content
    if isinstance(result, dict) and result.get("error"):
        return str(result["error"])
    return None


def _verification(agent: str, used: list[tuple[str, str | None]]) -> str:
    """Bilan factuel des outils appelés, ajouté par le code au rapport d'un sous-agent."""
    head = "--- Vérification automatique (produite par le code, pas par l'agent) ---\n"
    if not used:
        return (head + f"{agent} n'a appelé AUCUN outil : ce rapport ne s'appuie sur aucune donnée, et "
                "aucune action qu'il décrit n'a eu lieu.")
    lines = []
    for name in dict.fromkeys(n for n, _ in used):
        errors = [e for n, e in used if n == name and e is not None]
        ok = sum(1 for n, e in used if n == name and e is None)
        line = f"- {name} : {ok} réussi(s), {len(errors)} échec(s)"
        if errors:
            line += f" ; dernière erreur : {errors[-1][:200]}"
        lines.append(line)
    return head + f"Outils réellement appelés par {agent} :\n" + "\n".join(lines)


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _tokens(usage) -> dict:
    details = _get(usage, "prompt_tokens_details") or {}  # absent des réponses sans cache
    return {"input": _get(usage, "prompt_tokens") or 0,
            "cached_input": _get(details, "cached_tokens") or 0,
            "output": _get(usage, "completion_tokens") or 0}


def _split(content) -> tuple[list[str], list[str]]:
    """Sépare le raisonnement (chunks thinking) du texte. content : str ou liste de chunks."""
    if content is None:
        return [], []
    if isinstance(content, str):
        return [], [content] if content.strip() else []
    thinking, texts = [], []
    for chunk in content:
        if _get(chunk, "type") == "thinking":
            thinking.append("".join(_get(t, "text", "") for t in _get(chunk, "thinking") or []))
        elif _get(chunk, "type") == "text" and _get(chunk, "text", "").strip():
            texts.append(_get(chunk, "text"))
    return [t for t in thinking if t], texts


def _loggable(content) -> Any:
    """Résultat d'outil tel qu'enregistré : complet, images remplacées par un repère, JSON décodé."""
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return [c["text"] if c["type"] == "text" else "<image>" for c in content]
