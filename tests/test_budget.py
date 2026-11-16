"""Budget de session (agents/runner.py Session, délégation de l'orchestrateur), appels LLM simulés.

Chaque appel simulé coûte exactement COST $ (tokens d'entrée au tarif de MODEL, sans sortie).
"""

import json
from types import SimpleNamespace

import pytest

from agents import orchestrator
from agents.runner import PRICES, Agent, Session

MODEL = "mistral/mistral-large-4-0"  # pas de raisonnement à négocier : aucun import de LiteLLM
TOKENS = 100_000  # 0.07 $ : l'Evaluator (15 tours au plus) atteint sa limite avant ses tours
COST = TOKENS * PRICES[MODEL]["input"] / 1e6


class _Msg(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        out = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                 for c in self.tool_calls]
        return out


def _response(text=None, tool=None, args=None):
    calls = None
    if tool:
        calls = [SimpleNamespace(id="call_1", function=SimpleNamespace(name=tool, arguments=json.dumps(args or {})))]
    msg = _Msg(content=text, tool_calls=calls, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tool else "stop")],
                           usage={"prompt_tokens": TOKENS, "completion_tokens": 0})


class FakeLLM:
    """Orchestrateur : délègue `delegations` fois à l'Evaluator, puis rend son rapport.
    Sous-agent : appelle un outil à chaque tour, sans jamais conclure de lui-même."""

    def __init__(self, delegations=2):
        self.delegations = delegations
        self.calls = []

    def __call__(self, messages, tools=(), tool_choice="auto", **kwargs):
        orch = any(t["function"]["name"] == "ask_evaluator" for t in tools)
        self.calls.append("orchestrator" if orch else "sub")
        if tool_choice == "none":
            return _response(text="rapport forcé")
        if orch:
            asked = sum(1 for m in messages if m.get("role") == "tool")
            if asked < self.delegations:
                return _response(tool="ask_evaluator", args={"task": "évalue"})
            return _response(text="rapport final de l'orchestrateur")
        return _response(tool="list_experiments")


def _session(tmp_path, budget, llm=None):
    return Session(budget, completion=llm or FakeLLM(), run_dir=tmp_path / "run", verbose=False, model=MODEL)


@pytest.mark.parametrize("budget, reserve", [(1.0, 0.15), (0.2, 0.05), (0.06, 0.03), (10.0, 1.5)])
def test_report_reserve(tmp_path, budget, reserve):
    """15 % du budget, au moins 0.05 $, jamais plus de la moitié du budget."""
    assert _session(tmp_path, budget).reserve_usd == pytest.approx(reserve)


def test_explicit_reserve(tmp_path):
    s = Session(1.0, completion=FakeLLM(), run_dir=tmp_path / "run", verbose=False, model=MODEL,
                report_reserve_usd=0.4)
    assert s.reserve_usd == 0.4


def test_reserve_reached_boundary(tmp_path):
    s = _session(tmp_path, 1.0)
    s.spent = 0.849
    assert not s.reserve_reached
    s.spent = 0.85
    assert s.reserve_reached


def test_delegation_refused_once_reserve_reached(tmp_path, lab):
    s = _session(tmp_path, 1.0)
    s.spent = 0.9
    ask = next(t for t in orchestrator.make(lab, s).tools if t.name == "ask_evaluator")
    out = json.loads(ask.call({"task": "évalue"}))
    assert "budget de délégation épuisé" in out["error"]
    assert s.calls == 0, "aucun agent lancé"


def test_sub_agent_stops_before_reserve_and_report_still_written(tmp_path, lab):
    budget = 1.0
    llm = FakeLLM(delegations=2)
    s = _session(tmp_path, budget, llm)
    report = s.run(orchestrator.make(lab, s), "objectif")

    # Rapport rendu de lui-même ou au dernier tour sans outils, selon ce qui reste de la réserve
    assert report in ("rapport final de l'orchestrateur", "rapport forcé")
    # L'Evaluator a consommé sa part puis s'est arrêté sur le budget, avec un dernier rapport
    ends = {st["agent"]: st for st in s.trace["steps"] if st["type"] == "end"}
    assert ends["evaluator"]["arret"] == "budget"
    sub_spent = ends["evaluator"]["cost_usd"]
    assert budget - s.reserve_usd <= sub_spent <= budget - s.reserve_usd + COST + 1e-9
    # La 2e délégation a été refusée sans lancer d'agent : un seul appel de sous-agent
    assert s.calls == 2
    refused = [st for st in s.trace["steps"] if st["type"] == "tool" and st["agent"] == "orchestrator"]
    assert "budget de délégation épuisé" in refused[-1]["result"]["error"]
    # Budget total tenu à un appel près (la limite s'estime au coût du dernier appel) : la
    # réserve a payé le tour de la délégation refusée et le rapport de l'orchestrateur
    assert s.spent <= budget + COST + 1e-9
    assert llm.calls[-1] == "orchestrator"
    assert ends["orchestrator"]["final"] == report


def test_orchestrator_concludes_at_budget(tmp_path):
    """Sans délégation, un agent seul s'arrête au budget et rédige un dernier rapport sans outil."""
    llm = lambda messages, tool_choice="auto", **k: (_response(text="fin") if tool_choice == "none"
                                                     else _response(tool="noop"))
    s = _session(tmp_path, 0.1, llm)
    agent = Agent(name="seul", system="x", max_turns=1000, model=MODEL)
    assert s.run(agent, "tâche") == "fin"
    assert s.spent <= 0.1 + COST + 1e-9
    assert [st for st in s.trace["steps"] if st["type"] == "end"][0]["arret"] == "budget"
