"""Boucle tool use commune aux quatre agents.

Un `Agent` = un prompt système + une liste d'outils + un modèle. Une `Session` exécute des agents
via LiteLLM (chat completions + function calling au format OpenAI, traduit vers Mistral, OpenAI,
Anthropic ou Google selon le préfixe du modèle) avec un budget partagé et journalise tout
le pipeline dans runs/<NNN>_<horodatage>/ (à la racine du projet) : pipeline.json (réécrit à chaque
étape), plus report.md pour le rapport final.

pipeline.json = {"run": {...paramètres, statut, coût, tokens...}, "agents": {nom: {system,
tools}}, "steps": [...]}, les étapes dans l'ordre chronologique. Chaque étape porte `call` (n°
d'appel d'agent) et `parent` (appel qui l'a délégué), d'où l'arbre orchestrateur → sous-agents.
Types d'étape :
- start : un agent reçoit une tâche ;
- turn : une réponse du modèle (raisonnement, texte, outils demandés, tokens, coût cumulé) ;
- tool : un appel d'outil (paramètres, résultat complet, sortie console, durée, erreur) ;
- user : message ajouté par le code avant le tour suivant (images d'outils, relance de fin) ;
- end : rapport final de l'agent ;
- error : exception inattendue (avec traceback).
Le contexte envoyé au modèle à chaque tour se reconstitue donc exactement : system + tools de
l'agent, puis task, puis les étapes turn/tool/user de cet appel. Les images renvoyées par les
outils sont enregistrées dans images/ et référencées par leur chemin.

Chaque appel d'agent repart d'une conversation vide : l'état durable (dataset, expériences,
production) vit sur disque, pas dans les conversations.
"""

import base64
import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from video_source.annotators import MODELS
from video_source.config import LAB_DIR

RUNS_DIR = LAB_DIR / "runs"

# Modèles au format LiteLLM « fournisseur/identifiant », épinglés (pas d'alias -latest) pour qu'un
# changement de modèle côté fournisseur ne modifie pas le comportement en silence.
LARGE = "mistral/mistral-large-4-0"   # Mistral Large 4 (v26.10) : raisonnement, vision, 1M de contexte
SMALL = "mistral/mistral-small-2603"  # Mistral Small 4 (v26.03) : raisonnement, vision, 256K
# $/million de tokens : tarifs relevés à la main (table des modèles du banc des annotateurs,
# octobre 2026), prioritaires sur ceux de LiteLLM, parfois en retard. cached_input : tokens
# d'entrée servis depuis le cache, quand l'API les signale et que le fournisseur publie un prix.
PRICES = {f"{provider}/{model}": p for model, _, _, p, provider in MODELS}
# Effort de raisonnement des modèles qui ne prennent pas celui de l'agent (None : aucun).
REASONING_BY_MODEL = {"mistral/mistral-large-2512": None}  # Mistral Large 3 : pas de raisonnement


def _next_run_number() -> int:
    """Numéro du prochain run : 1 + le plus grand préfixe NNN_ des dossiers de runs/."""
    nums = [int(p.name.split("_")[0]) for p in RUNS_DIR.glob("[0-9][0-9][0-9]_*") if p.is_dir()]
    return max(nums, default=0) + 1


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
    reasoning: str = "high"  # reasoning_effort, traduit par LiteLLM pour chaque fournisseur


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


def price(model: str) -> dict:
    """Tarif d'un modèle en $/million de tokens : PRICES, sinon la table de LiteLLM. Sans tarif, le
    budget ne peut pas être tenu : erreur avant tout appel."""
    if model in PRICES:
        return PRICES[model]
    import litellm
    info = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/", 1)[-1]) or {}
    if not (info.get("input_cost_per_token") and info.get("output_cost_per_token")):
        raise ValueError(f"tarif inconnu pour {model} : l'ajouter à PRICES (agents/runner.py)")
    out = {"input": info["input_cost_per_token"] * 1e6, "output": info["output_cost_per_token"] * 1e6}
    if info.get("cache_read_input_token_cost"):
        out["cached_input"] = info["cache_read_input_token_cost"] * 1e6
    return out


class Session:
    def __init__(self, budget_usd: float, completion=None, run_dir: Path | None = None, verbose: bool = True,
                 params: dict | None = None, report_reserve_usd: float | None = None, model: str | None = None):
        if completion is None:
            import litellm
            litellm.suppress_debug_info = True
            completion = litellm.completion  # clés lues dans l'environnement (.env : MISTRAL_API_KEY…)
        self.completion = completion
        self.model = model  # même modèle pour tous les agents ; None : celui de chaque agent
        if model:
            price(model)
        self.budget_usd = budget_usd
        # Part du budget réservée à l'orchestrateur pour conclure : les sous-agents s'arrêtent avant,
        # et l'orchestrateur ne peut plus déléguer une fois la réserve entamée (run 004 : coupé
        # sans rapport final).
        self.reserve_usd = (report_reserve_usd if report_reserve_usd is not None
                            else min(max(0.05, 0.15 * budget_usd), 0.5 * budget_usd))
        self.spent = 0.0
        self.calls = 0
        self.verbose = verbose
        self.run_dir = run_dir or RUNS_DIR / f"{_next_run_number():03d}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._stack: list[int] = []  # appels d'agent en cours (le dernier est le plus profond)
        self.trace = {
            "run": {"status": "running", "started": _now(), "ended": None,
                    "budget_usd": budget_usd, "reserve_rapport_usd": round(self.reserve_usd, 4),
                    **(params or {}), "cost_usd": 0.0,
                    "tokens": {"input": 0, "cached_input": 0, "output": 0},
                    "agent_calls": 0, "report": None, "error": None},
            "agents": {},  # nom -> {system, tools} : ce que chaque agent reçoit en plus de ses messages
            "steps": [],
        }
        self._save()

    @property
    def reserve_reached(self) -> bool:
        """Le budget hors réserve est consommé : plus de délégation, place au rapport final."""
        return self.spent >= self.budget_usd - self.reserve_usd

    @staticmethod
    def _cost(model: str, tokens: dict) -> float:
        p = price(model)
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

    def _save_image(self, call: int, turn: int, k: int, source: dict) -> str:
        """Enregistre une image renvoyée par un outil ; renvoie son chemin relatif au dossier du run."""
        ext = {"image/jpeg": "jpg", "image/png": "png"}.get(source["media_type"], "bin")
        rel = f"images/call{call:03d}_turn{turn:02d}_{k:02d}.{ext}"
        (self.run_dir / "images").mkdir(exist_ok=True)
        (self.run_dir / rel).write_bytes(base64.b64decode(source["data"]))
        return rel

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

        model = self.model or agent.model
        price(model)  # tarif inconnu : erreur avant le premier appel
        effort = REASONING_BY_MODEL.get(model, agent.reasoning)
        # Effort transmis tel quel : la table de capacités de LiteLLM a du retard (zai-glm-5-3-flash
        # y figure sans raisonnement et l'appel serait refusé) ; c'est l'API du fournisseur qui juge.
        reasoning = {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]} if effort else {}
        tool_schemas = [t.schema() for t in agent.tools]
        # Prompt système et schémas d'outils : fixes pendant un run, enregistrés une fois par agent.
        self.trace["agents"].setdefault(agent.name, {"system": agent.system, "tools": tool_schemas})
        step("start", task=task, max_turns=agent.max_turns, model=model, reasoning=effort)
        self._say(f"\n=== {agent.name} ===\n{task}")
        tools = {t.name: t for t in agent.tools}
        n_images = 0
        messages = [{"role": "system", "content": agent.system}, {"role": "user", "content": task}]
        final = ""
        used: list[tuple[str, str | None]] = []  # (outil, erreur ou None) pour la vérification
        # Les sous-agents s'arrêtent à budget - réserve ; l'orchestrateur peut aller jusqu'au budget.
        limit = self.budget_usd if len(self._stack) == 1 else self.budget_usd - self.reserve_usd

        last_cost = 0.0  # coût du dernier appel : estimation du prochain

        def complete(turn: int, tool_choice: str):
            nonlocal last_cost
            t0 = time.perf_counter()
            response = self.completion(
                model=model,
                max_tokens=16000,
                messages=messages,
                tools=tool_schemas,
                tool_choice=tool_choice,
                **reasoning,
            )
            choice = response.choices[0]
            msg = choice.message
            tokens = _tokens(response.usage)
            cost = self._cost(model, tokens)
            self.spent += cost
            last_cost = cost
            for k, v in tokens.items():
                self.trace["run"]["tokens"][k] += v
            # Message complet, raisonnement compris : LiteLLM le rejoue si le fournisseur l'exige
            # (blocs signés d'Anthropic) et le retire sinon (Mistral refuse ces champs en entrée).
            messages.append(msg.model_dump(exclude_none=True))
            thinking, texts = _split(msg)
            calls = msg.tool_calls or []
            for t in texts:
                self._say(f"[{agent.name}] {t}")
            step("turn", turn=turn, stop=choice.finish_reason, thinking=thinking, text=texts,
                 tool_calls=[{"name": c.function.name, "input": c.function.arguments} for c in calls],
                 tokens=tokens, turn_cost_usd=round(cost, 5), cost_usd=round(self.spent, 4),
                 duration_s=round(time.perf_counter() - t0, 2))
            final = "\n\n".join(texts)
            if choice.finish_reason == "length":
                final += "\n\n[réponse tronquée : limite de tokens atteinte]"
            return calls, final

        def add_user(content: list[dict], logged: list[str]) -> None:
            """Message ajouté par le code : fusionné au précédent s'il vient déjà de l'utilisateur
            (pas deux messages user d'affilée), et journalisé car il fait partie du contexte."""
            last = messages[-1]
            if isinstance(last, dict) and last.get("role") == "user" and isinstance(last["content"], list):
                last["content"] += content
            else:
                messages.append({"role": "user", "content": content})
            step("user", content=logged)

        def last_report(turn: int, reason: str) -> str:
            """Dernier tour sans outils : l'agent rédige son rapport au lieu d'être coupé net."""
            text = (f"[{reason} Tu ne peux plus appeler d'outil. Rédige maintenant ton rapport final, "
                    "uniquement à partir de ce que tes outils ont renvoyé, en signalant ce qui reste à faire.]")
            add_user([{"type": "text", "text": text}], [text])
            _, final = complete(turn, "none")
            return final or f"Agent arrêté ({reason}) sans rapport."

        try:
            for turn in range(1, agent.max_turns + 1):
                calls, final = complete(turn, "auto")
                if not calls:
                    break

                images, image_paths = [], []  # l'API n'accepte les images que dans un message user
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
                    paths = []
                    if isinstance(content, list):
                        imgs = [b for b in content if b["type"] == "image"]
                        images += [{"type": "image_url", "image_url": {
                                       "url": f"data:{b['source']['media_type']};base64,{b['source']['data']}"}}
                                   for b in imgs]
                        for b in imgs:
                            n_images += 1
                            paths.append(self._save_image(call, turn, n_images, b["source"]))
                        image_paths += paths
                        text = "\n".join(b["text"] for b in content if b["type"] == "text")
                        if imgs:
                            text += f"\n[{len(imgs)} image(s) dans le message suivant]"
                    else:
                        text = content
                    messages.append({"role": "tool", "name": name, "content": text, "tool_call_id": c.id})
                    result = _loggable(content, paths)
                    used.append((name, _failure(content, result, tb)))
                    step("tool", tool=name, input=args, result=result, error=tb is not None,
                         traceback=tb, stdout=capture.buf.getvalue() or None,
                         duration_s=round(time.perf_counter() - t0, 2))

                follow_up, logged = list(images), [f"<image {p}>" for p in image_paths]
                if follow_up:
                    add_user(follow_up, logged)
                # Un tour de plus dépasserait la limite (estimé au coût du dernier) : on conclut.
                if self.spent + last_cost >= limit:
                    final = last_report(turn + 1, f"Budget atteint ({self.spent:.2f}/{limit:.2f} $).")
                    break
                remaining = agent.max_turns - turn
                if remaining == 0:
                    final = last_report(turn + 1, "Nombre maximal de tours atteint.")
                    break
                if remaining <= 2 or self.spent >= 0.8 * limit:
                    note = (f"[{remaining} tours restants, {self.spent:.2f}/{limit:.2f} $ dépensés sur ta "
                            "part du budget. Termine maintenant et rends ton rapport.]")
                    add_user([{"type": "text", "text": note}], [note])
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


def _split(msg) -> tuple[list[str], list[str]]:
    """Raisonnement et texte d'une réponse LiteLLM (reasoning_content, quel que soit le fournisseur)."""
    thinking, content = _get(msg, "reasoning_content"), _get(msg, "content")
    return ([thinking] if thinking and thinking.strip() else []), ([content] if content and content.strip() else [])


def _loggable(content, image_paths: list[str]) -> Any:
    """Résultat d'outil tel qu'enregistré : complet, images remplacées par leur fichier, JSON décodé."""
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    paths = iter(image_paths)
    return [c["text"] if c["type"] == "text" else f"<image {next(paths)}>" for c in content]
