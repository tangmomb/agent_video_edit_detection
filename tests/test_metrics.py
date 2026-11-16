"""Métriques et choix du seuil (model_training_pipeline/metrics.py)."""

import numpy as np
import pytest

from model_training_pipeline import metrics


def _best_recall_brute_force(y, probs, target):
    """Meilleur rappel parmi les seuils (valeurs de probs) de précision >= target, par force brute."""
    best = None
    for t in np.unique(probs):
        op = metrics.operating_point(y, probs, t)
        if op["tp"] and op["precision"] >= target and (best is None or op["recall"] > best):
            best = op["recall"]
    return best


def test_operating_point_counts():
    y = np.array([1, 1, 0, 0, 1])
    probs = np.array([0.9, 0.4, 0.8, 0.1, 0.6])
    op = metrics.operating_point(y, probs, 0.5)
    assert (op["tp"], op["fp"], op["fn"], op["tn"]) == (2, 1, 1, 1)
    assert op["precision"] == pytest.approx(2 / 3)
    assert op["recall"] == pytest.approx(2 / 3)


def test_target_reached_on_separable_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    op = metrics.recall_at_precision(y, probs, 0.95)
    assert op is not None
    assert op["precision"] == 1.0 and op["recall"] == 1.0
    assert 0.3 < op["threshold"] <= 0.7


@pytest.mark.parametrize("target", metrics.PRECISION_TARGETS)
@pytest.mark.parametrize("seed", range(5))
def test_threshold_keeps_target_with_best_recall(target, seed):
    """Le seuil choisi tient la précision visée et donne le meilleur rappel possible."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 400)
    probs = np.clip(rng.normal(0.35 + 0.3 * y, 0.15), 0, 1)
    expected = _best_recall_brute_force(y, probs, target)
    op = metrics.recall_at_precision(y, probs, target)
    if expected is None:
        assert op is None
    else:
        assert op["precision"] >= target
        assert op["recall"] == pytest.approx(expected)


def test_target_not_reached():
    """Une négative en tête et des classes alternées : la précision ne dépasse jamais 0.5."""
    y = np.array([0, 1] * 20)
    probs = np.linspace(1, 0, len(y))
    assert metrics.recall_at_precision(y, probs, 0.9) is None
    assert metrics.summary(y, probs)["rappel_a_precision_0.9"] == "précision non atteinte"


def test_no_positive():
    y = np.zeros(10, dtype=int)
    probs = np.linspace(0, 1, 10)
    assert metrics.recall_at_precision(y, probs, 0.9) is None
    assert metrics.summary(y, probs)["ap"] is None


def _grouped_data(seed=0, chains=30, per_chain=20):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"youtube:c{k}" for k in range(chains)], per_chain)
    shift = np.repeat(rng.normal(0, 0.1, chains), per_chain)  # habillage propre à chaque chaîne
    y = rng.integers(0, 2, len(groups))
    probs = np.clip(rng.normal(0.3 + 0.4 * y, 0.15) + shift, 0, 1)
    return y, probs, groups


def test_robust_threshold_is_reproducible():
    """Même entrée, même seuil prudent : le bootstrap a sa graine fixe."""
    y, probs, groups = _grouped_data()
    a = metrics.robust_threshold(y, probs, groups, 0.9, draw_groups=10)
    b = metrics.robust_threshold(y, probs, groups, 0.9, draw_groups=10)
    assert a == b
    assert metrics.summary(y, probs, groups, 10) == metrics.summary(y, probs, groups, 10)


def test_robust_threshold_not_below_plain_threshold():
    """Le seuil prudent (centile haut des seuils tirés) n'est pas plus bas que le seuil simple."""
    y, probs, groups = _grouped_data(seed=3)
    plain = metrics.recall_at_precision(y, probs, 0.9)
    prudent = metrics.recall_at_precision(y, probs, 0.9, groups, draw_groups=10)
    assert prudent["threshold"] >= plain["threshold"]
    assert prudent["recall"] <= plain["recall"]
    assert prudent["seuil_prudent"] == {"centile": metrics.BOOTSTRAP_QUANTILE, "chaines_par_tirage": 10}


def test_group_ap_ignores_single_class_chains():
    y = np.array([1, 0, 1, 0, 1, 1])
    probs = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.6])
    groups = np.array(["a", "a", "b", "b", "c", "c"])  # c n'a que des positives : ignorée
    # a : classement parfait (AP 1) ; b : la positive sous la négative (AP 0.5)
    assert metrics.group_ap(y, probs, groups) == pytest.approx(0.75)
    assert metrics.group_ap(np.ones(4, dtype=int), probs[:4], groups[:4]) is None
