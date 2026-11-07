"""Métriques du classifieur binaire « la frame contient un ajout de montage ».

Métrique principale : rappel à précision fixée. Le seuil est choisi sur val (le plus bas qui
garde la précision >= cible, donc le meilleur rappel), puis appliqué tel quel sur test.

Seuil prudent (groups donnés, une chaîne YouTube par image) : le seuil qui tient la précision sur
un jeu de chaînes ne la tient pas sur d'autres, car chaque chaîne a son habillage et décale les
scores (banc b1 : précision de 0.53 à 1.0 sur des chaînes cachées pour 95 % visés). On tire donc
des jeux de chaînes au hasard (bootstrap par chaîne), de la taille du test, on calcule le seuil de
chacun et on garde un centile haut : la précision visée tient sur la plupart des jeux, au prix
d'un peu de rappel.

AP par chaîne (groups donnés) : moyenne des AP de chaque chaîne qui a les deux classes. Elle
mesure le classement des images à l'intérieur d'une chaîne, sans qu'une grosse chaîne pèse plus
que les autres ; c'est le critère de choix du modèle de production. Banc b2 (384 expériences) :
choisir par rappel val à 95 % gardait souvent CLIP seul (0.53 de rappel idéal moyen sur des
chaînes cachées), choisir par AP par chaîne prend le meilleur modèle caché dans 19 runs sur 30
au lieu de 5 (0.63), avec une précision cachée plus proche de la cible (0.94 au lieu de 0.88).
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

PRECISION_TARGETS = (0.9, 0.95, 0.99)
# 0.95 jusqu'au 03/11/2026 : à 90 %, exp_151 passait de 85,9 à 91,3 % de rappel val (40 -> 99 FP).
DEFAULT_PRECISION = 0.9
BOOTSTRAP_DRAWS = 200
# Centile des seuils tirés : plus haut = plus prudent. 80 jusqu'au 07/11/2026, avec des jeux de
# la taille du val (311 chaînes) : le seuil tenait sur ces gros jeux, pas sur les 77 chaînes du
# test (run 019 : 3 tests échoués sur split_008, précision test 0,868 à 0,893 pour 0,911 sur val).
# Tirés à la taille du test et au centile 90, 4 candidats sur 5 de split_008 passaient (exp_394 :
# 0,921), pour environ 1 point de rappel val en moins.
BOOTSTRAP_QUANTILE = 90


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


def prudent_method(chains_per_draw: int) -> dict:
    """Réglage du seuil prudent, gardé avec chaque point de fonctionnement : un seuil calculé avec
    un autre réglage est périmé (Lab._val_at le recalcule)."""
    return {"centile": BOOTSTRAP_QUANTILE, "chaines_par_tirage": int(chains_per_draw)}


def robust_threshold(y: np.ndarray, probs: np.ndarray, groups: np.ndarray, target: float,
                     draw_groups: int | None = None) -> float:
    """Seuil prudent : centile BOOTSTRAP_QUANTILE des seuils de jeux de chaînes tirés avec remise,
    de draw_groups chaînes chacun (la taille du test du split ; à défaut, autant que dans groups).
    Un petit jeu varie plus qu'un gros : tirés à la taille du val, les jeux sous-estimaient la
    variation sur le test."""
    groups = np.asarray(groups)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    size = int(draw_groups or len(members))
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = np.concatenate([members[i] for i in rng.integers(len(members), size=size)])
        draws.append(_threshold(y[idx], probs[idx], target))
    # method="higher" : jamais d'interpolation, qui donnerait nan entre deux seuils infinis
    return float(np.percentile(draws, BOOTSTRAP_QUANTILE, method="higher"))


def recall_at_precision(y: np.ndarray, probs: np.ndarray, target: float,
                        groups: np.ndarray | None = None, draw_groups: int | None = None) -> dict | None:
    """Point de fonctionnement de meilleur rappel parmi ceux de précision >= target ; avec groups,
    au seuil prudent (robust_threshold, réglage noté dans seuil_prudent). None si la précision
    visée n'est pas atteinte."""
    t = (_threshold(y, probs, target) if groups is None
         else robust_threshold(y, probs, groups, target, draw_groups))
    if not np.isfinite(t):
        return None
    op = operating_point(y, probs, t)
    if groups is not None:
        op["seuil_prudent"] = prudent_method(draw_groups or len(np.unique(groups)))
    return op if op["recall"] > 0 else None


def group_ap(y: np.ndarray, probs: np.ndarray, groups: np.ndarray) -> float | None:
    """AP moyenne des groupes (chaînes) qui ont les deux classes ; None s'il n'y en a aucun."""
    groups = np.asarray(groups)
    aps = [average_precision_score(y[m], probs[m]) for m in (groups == g for g in np.unique(groups))
           if 0 < y[m].sum() < m.sum()]
    return float(np.mean(aps)) if aps else None


def summary(y: np.ndarray, probs: np.ndarray, groups: np.ndarray | None = None,
            draw_groups: int | None = None) -> dict:
    out = {"auc": roc_auc_score(y, probs) if len(set(y)) > 1 else None,
           "ap": average_precision_score(y, probs) if y.any() else None,
           "positifs": int(y.sum()), "images_extraites_des_clips": len(y)}
    if groups is not None:
        out["ap_par_chaine"] = group_ap(y, probs, groups)
    for p in PRECISION_TARGETS:
        out[f"rappel_a_precision_{p}"] = recall_at_precision(y, probs, p, groups, draw_groups) or "précision non atteinte"
    return out
