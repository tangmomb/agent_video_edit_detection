"""Splits par chaîne YouTube (Lab.make_split, _stratified, _folds).

Le test le plus important du projet : une chaîne garde le même habillage d'une vidéo à l'autre.
Si elle se retrouvait des deux côtés (dev et test, ou deux plis de validation), le modèle
reconnaîtrait la chaîne au lieu de l'ajout de montage, et les scores seraient faux.
"""

import json
from collections import defaultdict

import pytest

from model_training_pipeline.lab import FOLDS, _stratified


def _fake_dataset(lab, chains: int, videos_per_chain: int = 3):
    """Fausses vidéos : chaque chaîne a plusieurs vidéos, d'orientations et d'étiquettes variées."""
    rows, metas = [], {}
    for c in range(chains):
        for v in range(videos_per_chain):
            sid = f"yt_c{c}_v{v}"
            portrait = (c + v) % 3 == 0
            metas[sid] = {"channel_id": f"chaine{c}",
                          "video": {"width": 360 if portrait else 640, "height": 640 if portrait else 360}}
            empty = c % 4 == 1 and v == 0  # quelques vidéos sans aucun ajout
            for frame in range(4):
                rows.append({"sample_id": sid, "source": f"youtube:chaine{c}",
                             "label_any": 0 if empty else frame % 2})
    lab._rows = rows
    lab._ann = metas.__getitem__
    return rows


def _parts_by_chain(lab, split_id):
    parts = defaultdict(set)
    for sid, part in lab._split_map(split_id).items():
        parts[lab._ann(sid)["channel_id"]].add(part)
    return parts


@pytest.mark.parametrize("chains", [3, 4, 12])
@pytest.mark.parametrize("seed", range(5))
def test_chain_never_in_dev_and_test(lab, chains, seed):
    _fake_dataset(lab, chains)
    split_id = lab.make_split("test", test_ratio=0.25, seed=seed)["split_id"]
    parts = _parts_by_chain(lab, split_id)
    assert len(parts) == chains, "chaque chaîne doit être dans le split"
    mixed = {c: p for c, p in parts.items() if len(p) > 1}
    assert not mixed, f"chaînes à la fois en dev et en test : {mixed}"
    assert {p for ps in parts.values() for p in ps} == {"dev", "test"}


@pytest.mark.parametrize("seed", range(5))
def test_validation_folds_split_by_chain(lab, seed):
    """Validation croisée sur dev : chaque chaîne de dev dans un seul pli, aucune chaîne de test."""
    rows = _fake_dataset(lab, chains=12)
    split_id = lab.make_split("test", test_ratio=0.25, seed=seed)["split_id"]
    folds = lab._folds(split_id)
    parts = _parts_by_chain(lab, split_id)
    dev = {f"youtube:{c}" for c, p in parts.items() if p == {"dev"}}
    assert set(folds) == dev, "les plis couvrent exactement les chaînes de dev"
    assert len(set(folds.values())) == min(FOLDS, len(dev))
    # Comme train_model : une image de val (pli k) n'a aucune chaîne en commun avec l'entraînement
    dev_rows = [rows[i] for i in lab._idx("dev", split_id)]
    for k in set(folds.values()):
        val_chains = {r["source"] for r in dev_rows if folds[r["source"]] == k}
        train_chains = {r["source"] for r in dev_rows if folds[r["source"]] != k}
        assert val_chains and not val_chains & train_chains


def test_folds_are_frozen_in_split_file(lab, lab_dirs):
    """Les plis sont enregistrés avec le split : ils ne bougent plus si les étiquettes changent."""
    _fake_dataset(lab, chains=8)
    split_id = lab.make_split("test", seed=1)["split_id"]
    saved = json.loads((lab_dirs["splits"] / f"{split_id}.json").read_text(encoding="utf-8"))
    assert saved["folds"] == lab._folds(split_id)


def test_split_is_reproducible(lab, lab_dirs):
    _fake_dataset(lab, chains=12)
    a = lab.make_split("a", seed=7)["split_id"]
    b = lab.make_split("b", seed=7)["split_id"]
    read = lambda s: json.loads((lab_dirs["splits"] / f"{s}.json").read_text(encoding="utf-8"))
    assert read(a)["clips"] == read(b)["clips"]
    assert read(a)["folds"] == read(b)["folds"]


def test_split_needs_three_chains(lab):
    _fake_dataset(lab, chains=2)
    with pytest.raises(ValueError, match="3 chaînes"):
        lab.make_split("test")


@pytest.mark.parametrize("seed", range(10))
def test_stratified_assigns_each_chain_once(seed):
    clips = {f"youtube:c{k}": ({"orientation:portrait"} if k % 3 else {"orientation:paysage", "clip_vide"})
             for k in range(20)}
    assign = _stratified(clips, {"dev": 0.8, "test": 0.2}, seed)
    assert set(assign) == set(clips)
    assert sum(v == "test" for v in assign.values()) == 4
