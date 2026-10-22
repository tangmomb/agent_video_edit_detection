import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("EDIT_GENERATOR_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
SAMPLES_DIR = DATA_DIR / "videos_samples"
ASSETS_DIR = ROOT / "assets"


def ffmpeg_bin() -> str:
    custom = os.getenv("FFMPEG_BIN")
    if custom:
        return custom
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
