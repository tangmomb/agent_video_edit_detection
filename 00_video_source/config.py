import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Dossier de travail du labo : data/, model_results/ et runs/ (la racine du projet par défaut). Le
# banc d'agents (agents/bench.py) en donne un à chaque run pour ne jamais toucher aux vraies données.
LAB_DIR = Path(os.getenv("AGENT_LAB_DIR", ROOT))
DATA_DIR = Path(os.getenv("VIDEO_SOURCE_DATA_DIR", LAB_DIR / "data"))
SAMPLES_DIR = DATA_DIR / "videos_samples"
FRAMES_DIR = DATA_DIR / "videos_samples_frames"  # images extraites et embeddings (model_training_pipeline)


def ffmpeg_bin() -> str:
    custom = os.getenv("FFMPEG_BIN")
    if custom:
        return custom
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
