"""Evaluator (agents/evaluator.py) : il ne teste que candidat_a_tester et ne promeut rien."""

import json

import pytest

from agents import evaluator

ALLOWED_TOOLS = {"list_experiments", "analyze_errors", "show_examples", "final_test"}
PROMOTING = ("save_model", "select_production")


@pytest.fixture
def ev_lab(lab, lab_dirs):
    """Lab dont le candidat et le blocage sont fixés par le test ; final_test et la mise en
    production sont espionnés."""
    calls = {"final_test": [], "promotion": []}
    state = {"candidate": "exp_002", "blocked": None}
    lab.latest_split = lambda: "split_001"
    lab.candidate_to_test = lambda sid, precision=0.9: state["candidate"]
    lab.retest_blocked = lambda sid, precision=0.9: state["blocked"]
    lab.final_test = lambda eid, precision=0.9: calls["final_test"].append(eid) or {"statut": "réussi"}
    for name in PROMOTING:
        setattr(lab, name, lambda *a, _n=name, **k: calls["promotion"].append(_n))
    return lab, state, calls, lab_dirs["exp"]


def _final_test(lab):
    return next(t for t in evaluator.make(lab).tools if t.name == "final_test")


def test_tests_the_authorized_candidate(ev_lab):
    lab, _, calls, _ = ev_lab
    assert json.loads(_final_test(lab).call({"experiment_id": "exp_002"})) == {"statut": "réussi"}
    assert calls["final_test"] == ["exp_002"]


@pytest.mark.parametrize("other", ["exp_001", "exp_003", "", "exp_002 "])
def test_refuses_any_other_experiment(ev_lab, other):
    lab, _, calls, _ = ev_lab
    out = json.loads(_final_test(lab).call({"experiment_id": other}))
    assert "exp_002" in out["error"]
    assert calls["final_test"] == []


def test_refuses_when_no_candidate(ev_lab):
    lab, state, calls, _ = ev_lab
    state["candidate"] = None
    assert "error" in json.loads(_final_test(lab).call({"experiment_id": "exp_002"}))
    assert calls["final_test"] == []


def test_refuses_when_retest_blocked(ev_lab):
    lab, state, calls, _ = ev_lab
    state["blocked"] = "test déjà échoué sur split_001"
    out = json.loads(_final_test(lab).call({"experiment_id": "exp_002"}))
    assert out["error"] == "test déjà échoué sur split_001"
    assert calls["final_test"] == []


def test_has_no_promotion_tool():
    """Seuls ses quatre outils ; aucun ne met en production (même pas indirectement par son code)."""
    agent = evaluator.make(_Strict())
    assert {t.name for t in agent.tools} == ALLOWED_TOOLS
    for t in agent.tools:
        fn = t.fn
        assert getattr(fn, "__name__", "") not in PROMOTING
        names = set(getattr(getattr(fn, "__code__", None), "co_names", ()))
        assert not names & set(PROMOTING), f"{t.name} appelle {names & set(PROMOTING)}"


def test_final_test_never_promotes(ev_lab):
    lab, _, calls, exp_dir = ev_lab
    tool = _final_test(lab)
    for eid in ("exp_001", "exp_002", "exp_003"):
        tool.call({"experiment_id": eid})
    assert calls["promotion"] == []
    assert not (exp_dir / "production.json").exists()


class _Strict:
    """Faux Lab qui échoue au moindre appel de mise en production (et répond à tout le reste)."""

    def __getattr__(self, name):
        if name in PROMOTING:
            raise AssertionError(f"l'Evaluator accède à {name}")
        return lambda *a, **k: None
