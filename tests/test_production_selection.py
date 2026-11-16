"""Choix du modèle de production (Lab.select_production et _best_by_rule).

Règle actuelle :
1. splits du plus récent au plus ancien : le premier qui a un test réussi donne la production,
   choisie parmi ses seuls modèles réussis ;
2. aucun test réussi nulle part : la production actuelle est conservée si elle avait réussi le sien ;
3. sinon, un modèle du split le plus récent est promu, marqué non vérifié (non testé avant échoué).
Départage : AP par chaîne à AP_TIE près de la meilleure, puis meilleur rappel val.
"""

import itertools
import json

import pytest

from model_training_pipeline.lab import AP_TIE, _best_by_rule


def row(eid, ap_chain, recall, statut="non testé", ap=0.5, auc=0.5, ms=1.0):
    return {"experiment_id": eid, "ap_par_chaine": ap_chain, "val_recall": recall, "ap": ap, "auc": auc,
            "ms_par_frame": ms, "test": {"statut": statut}}


@pytest.fixture
def selector(lab, lab_dirs):
    """Lab dont les splits et leurs classements sont donnés par le test ; save_model enregistré."""
    state = {"ranked": {}, "saved": []}

    def save_model(eid, precision, note, test=None):
        state["saved"].append(eid)
        return {"experiment_id": eid, "test": test}

    lab._split_summaries = lambda: [{"split_id": s} for s in state["ranked"]]
    lab._ranked = lambda sid, precision=0.9: state["ranked"][sid]
    lab.save_model = save_model
    lab_dirs["exp"].mkdir(parents=True, exist_ok=True)
    state["lab"], state["exp"] = lab, lab_dirs["exp"]
    return state


def test_most_recent_split_with_passed_test_wins(selector):
    """Run 030 : split_017 sans test réussi, exp_711 réussi sur split_016 -> exp_711."""
    selector["ranked"] = {
        "split_015": [row("exp_666", 0.80, 0.70, "réussi")],
        "split_016": [row("exp_711", 0.78, 0.60, "réussi"), row("exp_700", 0.79, 0.65, "échoué")],
        "split_017": [row("exp_800", 0.90, 0.90, "non testé"), row("exp_801", 0.85, 0.80, "échoué")],
    }
    out = selector["lab"].select_production()
    assert out["split_id"] == "split_016" and out["dernier_split"] == "split_017"
    assert out["verifie_sur_test"] is True and out["production_conservee"] is False
    assert selector["saved"] == ["exp_711"]


def test_only_passed_models_promoted_when_one_passed(selector):
    """Un modèle non testé ou échoué, même bien meilleur sur val, ne passe pas devant un réussi."""
    selector["ranked"] = {"split_001": [
        row("exp_untested", 0.95, 0.95, "non testé"),
        row("exp_failed", 0.93, 0.92, "échoué"),
        row("exp_passed", 0.70, 0.50, "réussi"),
    ]}
    selector["lab"].select_production()
    assert selector["saved"] == ["exp_passed"]


def test_verified_production_kept_when_nothing_passed(selector):
    selector["ranked"] = {"split_002": [row("exp_new", 0.95, 0.95, "non testé")]}
    current = {"experiment_id": "exp_old", "test": {"statut": "réussi"}}
    (selector["exp"] / "production.json").write_text(json.dumps(current), encoding="utf-8")
    out = selector["lab"].select_production()
    assert out["production_conservee"] is True and out["verifie_sur_test"] is True
    assert out["production"]["experiment_id"] == "exp_old"
    assert selector["saved"] == []


@pytest.mark.parametrize("current_status", [None, "échoué", "non testé"])
def test_unverified_fallback_prefers_untested_over_failed(selector, current_status):
    selector["ranked"] = {
        "split_001": [row("exp_old_split", 0.99, 0.99, "non testé")],
        "split_002": [row("exp_failed", 0.95, 0.95, "échoué"), row("exp_untested", 0.80, 0.70, "non testé")],
    }
    if current_status:
        current = {"experiment_id": "exp_cur", "test": {"statut": current_status}}
        (selector["exp"] / "production.json").write_text(json.dumps(current), encoding="utf-8")
    out = selector["lab"].select_production()
    assert out["verifie_sur_test"] is False and out["split_id"] == "split_002"
    assert selector["saved"] == ["exp_untested"]


def test_no_split(selector):
    assert "error" in selector["lab"].select_production()
    assert selector["saved"] == []


# ------------------------------------------------------------------ départage


def test_ap_within_tie_then_best_recall():
    rows = [row("a", 0.800, 0.60), row("b", 0.800 - AP_TIE / 2, 0.70), row("c", 0.800 - 2 * AP_TIE, 0.99)]
    # c a le meilleur rappel mais son AP est hors de la marge : b gagne au rappel parmi a et b
    assert _best_by_rule(rows)["experiment_id"] == "b"


def test_full_tie_broken_by_ap_auc_then_speed():
    base = dict(ap_chain=0.8, recall=0.7)
    assert _best_by_rule([row("a", **base, ap=0.5), row("b", **base, ap=0.6)])["experiment_id"] == "b"
    assert _best_by_rule([row("a", **base, auc=0.9), row("b", **base, auc=0.8)])["experiment_id"] == "a"
    assert _best_by_rule([row("a", **base, ms=3.0), row("b", **base, ms=2.0)])["experiment_id"] == "b"


def test_missing_ap_counts_as_zero():
    assert _best_by_rule([row("a", None, 0.99), row("b", 0.5, 0.1)])["experiment_id"] == "b"
    assert _best_by_rule([]) is None


def test_rule_does_not_depend_on_order():
    rows = [row("a", 0.800, 0.60), row("b", 0.798, 0.70, ap=0.4), row("c", 0.798, 0.70, ap=0.6),
            row("d", 0.700, 0.99), row("e", None, 0.95)]
    winners = {_best_by_rule(list(p))["experiment_id"] for p in itertools.permutations(rows)}
    assert winners == {"c"}


def test_selection_does_not_depend_on_ranking_order(selector):
    rows = [row("exp_a", 0.80, 0.60, "réussi"), row("exp_b", 0.798, 0.70, "réussi"),
            row("exp_c", 0.95, 0.99, "échoué"), row("exp_d", 0.70, 0.99, "réussi")]
    for perm in itertools.permutations(rows):
        selector["ranked"] = {"split_001": list(perm)}
        selector["lab"].select_production()
    assert set(selector["saved"]) == {"exp_b"}
