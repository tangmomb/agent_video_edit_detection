"""Métriques du classifieur binaire « la frame contient un ajout de montage ».

Métrique principale : rappel à précision fixée. Le seuil est choisi sur val (le plus bas qui
garde la précision >= cible, donc le meilleur rappel), puis appliqué tel quel sur test.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

PRECISION_TARGETS = (0.9, 0.95, 0.99)
DEFAULT_PRECISION = 0.95


def operating_point(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = probs >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {"threshold": float(threshold), "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1) if tp + fp else 1.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def recall_at_precision(y: np.ndarray, probs: np.ndarray, target: float) -> dict | None:
    """Point de fonctionnement de meilleur rappel parmi ceux de précision >= target."""
    if not y.any():
        return None
    precision, recall, thresholds = precision_recall_curve(y, probs)
    ok = np.flatnonzero((precision[:-1] >= target) & (recall[:-1] > 0))
    if not len(ok):
        return None
    return operating_point(y, probs, thresholds[ok[np.argmax(recall[ok])]])


def summary(y: np.ndarray, probs: np.ndarray) -> dict:
    out = {"auc": roc_auc_score(y, probs) if len(set(y)) > 1 else None,
           "ap": average_precision_score(y, probs) if y.any() else None,
           "positifs": int(y.sum()), "images_extraites_des_clips": len(y)}
    for p in PRECISION_TARGETS:
        out[f"rappel_a_precision_{p}"] = recall_at_precision(y, probs, p) or "précision non atteinte"
    return out
