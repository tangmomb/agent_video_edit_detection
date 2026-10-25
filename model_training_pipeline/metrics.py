"""Métriques du classifieur binaire « la frame contient un ajout de montage ».

Métrique principale : rappel à précision fixée. Le seuil est choisi sur val (le plus bas qui
garde la précision >= cible, donc le meilleur rappel), puis appliqué tel quel sur test.

Seuil prudent (groups donnés, une chaîne YouTube par image) : le seuil qui tient la précision sur
un jeu de chaînes ne la tient pas sur d'autres, car chaque chaîne a son habillage et décale les
scores (banc b1 : précision de 0.53 à 1.0 sur des chaînes cachées pour 95 % visés). On tire donc
des jeux de chaînes au hasard (bootstrap par chaîne), on calcule le seuil de chacun et on garde
un centile haut : la précision visée tient sur la plupart des jeux, au prix d'un peu de rappel.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

PRECISION_TARGETS = (0.9, 0.95, 0.99)
DEFAULT_PRECISION = 0.95
BOOTSTRAP_DRAWS = 200
BOOTSTRAP_QUANTILE = 80  # centile des seuils tirés : plus haut = plus prudent


def operating_point(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = probs >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {"threshold": float(threshold), "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1) if tp + fp else 1.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _threshold(y: np.ndarray, probs: np.ndarray, target: float) -> float:
    """Seuil de meilleur rappel parmi ceux de précision >= target (inf si jamais atteinte)."""
    if not y.any():
        return np.inf
    precision, recall, thresholds = precision_recall_curve(y, probs)
    ok = np.flatnonzero((precision[:-1] >= target) & (recall[:-1] > 0))
    return float(thresholds[ok[np.argmax(recall[ok])]]) if len(ok) else np.inf


def robust_threshold(y: np.ndarray, probs: np.ndarray, groups: np.ndarray, target: float) -> float:
    """Seuil prudent : centile BOOTSTRAP_QUANTILE des seuils de jeux de chaînes tirés avec remise."""
    groups = np.asarray(groups)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        draws.append(_threshold(y[idx], probs[idx], target))
    # method="higher" : jamais d'interpolation, qui donnerait nan entre deux seuils infinis
    return float(np.percentile(draws, BOOTSTRAP_QUANTILE, method="higher"))


def recall_at_precision(y: np.ndarray, probs: np.ndarray, target: float,
                        groups: np.ndarray | None = None) -> dict | None:
    """Point de fonctionnement de meilleur rappel parmi ceux de précision >= target ; avec groups,
    au seuil prudent (robust_threshold). None si la précision visée n'est pas atteinte."""
    t = _threshold(y, probs, target) if groups is None else robust_threshold(y, probs, groups, target)
    if not np.isfinite(t):
        return None
    op = operating_point(y, probs, t)
    return op if op["recall"] > 0 else None


def summary(y: np.ndarray, probs: np.ndarray, groups: np.ndarray | None = None) -> dict:
    out = {"auc": roc_auc_score(y, probs) if len(set(y)) > 1 else None,
           "ap": average_precision_score(y, probs) if y.any() else None,
           "positifs": int(y.sum()), "images_extraites_des_clips": len(y)}
    for p in PRECISION_TARGETS:
        out[f"rappel_a_precision_{p}"] = recall_at_precision(y, probs, p, groups) or "précision non atteinte"
    return out
