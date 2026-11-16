"""Tests sans GPU, sans appel d'API et sans toucher aux vraies données.

Le dossier de travail du labo (AGENT_LAB_DIR) est redirigé vers un dossier temporaire AVANT tout
import du projet : video_source.config le lit à l'import, et load_dotenv ne remplace pas une
variable déjà définie. Chaque test qui écrit (splits, production.json) reçoit en plus ses propres
dossiers via la fixture lab_dirs.
"""

import os
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="agent_tests_"))
os.environ["AGENT_LAB_DIR"] = str(_SANDBOX)
os.environ["VIDEO_SOURCE_DATA_DIR"] = str(_SANDBOX / "data")
# Ni entraînement ni embeddings sur Modal pendant les tests.
os.environ.pop("TRAIN_GPU", None)
os.environ.pop("EMBED_GPU", None)

import pytest  # noqa: E402


@pytest.fixture
def lab_dirs(tmp_path, monkeypatch):
    """Dossiers propres à un test pour les splits, les expériences et les exclusions."""
    from model_training_pipeline import lab as lab_mod

    exp_dir, splits_dir = tmp_path / "model_results", tmp_path / "data" / "splits"
    monkeypatch.setattr(lab_mod, "EXP_DIR", exp_dir)
    monkeypatch.setattr(lab_mod, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(lab_mod, "EXCLUDED", tmp_path / "data" / "excluded.json")
    return {"exp": exp_dir, "splits": splits_dir}


@pytest.fixture
def lab(lab_dirs):
    """Un Lab vide, sur les dossiers du test."""
    from model_training_pipeline.lab import Lab

    return Lab()
