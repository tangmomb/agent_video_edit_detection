"""Construit le dataset de frames étiquetées à partir des échantillons editgen.

Chaque ligne de index.jsonl :
  frame, sample_id, t, source, split,
  visible      : ids des éléments pleinement visibles (opacité >= 0.9)
  partial      : ids en cours de fondu (0.05 < opacité < 0.9)
  visible_text : sous-ensemble textuel de visible (textes ajoutés)
  label_any    : 1 si au moins un élément pleinement visible
  label_text   : 1 si au moins un élément *textuel* pleinement visible
  ambiguous    : True si un élément est en fondu (exclu de l'entraînement)
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from editgen import annotations
from editgen.config import DATA_DIR, SAMPLES_DIR, ffmpeg_bin

TEXT_TYPES = {"keyword", "question", "subtitle", "lower_third"}
FRAMES_DIR = DATA_DIR / "frames"

FPS = 1.0  # frames extraites par seconde de vidéo


def source_key(ann: dict) -> str:
    s = ann["source"]
    return f"{s['kind']}:{s.get('video_id') or s.get('path')}"


def split_for(key: str, test_ratio: float = 0.2, val_ratio: float = 0.2) -> str:
    # Split par vidéo *source* : deux échantillons tirés du même clip restent du même côté.
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 1000
    if h < test_ratio * 1000:
        return "test"
    if h < (test_ratio + val_ratio) * 1000:
        return "val"
    return "train"


# Version de l'extraction, notée dans img/<sample_id>/.extraction : des frames extraites par une
# autre version (et leurs embeddings) sont refaites au prochain build().
EXTRACTION = "fps-round-up"


def extract(video: Path, out_dir: Path, fps: float, long_side: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = f"scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'"
    # round=up : la frame i est celle de l'instant i/fps, celui où on calcule son label. Avec le
    # défaut (round=near), ffmpeg prend la frame ~0,5 s plus tard : 7,5 % de labels faux près des
    # apparitions/disparitions d'ajouts (constaté sur s000013, octobre 2026).
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vf", f"fps={fps}:round=up,{scale}", "-q:v", "3", str(out_dir / "%05d.jpg")]
    subprocess.run(cmd, check=True)
    (out_dir / ".extraction").write_text(EXTRACTION)
    return sorted(out_dir.glob("*.jpg"))


def _stale(frames_dir: Path) -> bool:
    marker = frames_dir / ".extraction"
    return frames_dir.exists() and (not marker.exists() or marker.read_text() != EXTRACTION)


def _drop(out_dir: Path, sample_id: str) -> None:
    """Supprime les frames d'un échantillon et leurs embeddings en cache (données dérivées)."""
    shutil.rmtree(out_dir / "img" / sample_id)
    for f in (out_dir / "feat").glob(f"*/{sample_id}.npy"):
        f.unlink()


def label(ann: dict, t: float) -> dict:
    visible, partial = [], []
    for el in ann["elements"]:
        op = annotations.opacity_at(el, t)
        if op >= 0.9:
            visible.append(el["id"])
        elif op > 0.05:
            partial.append(el["id"])
    types = {el["id"]: el["type"] for el in ann["elements"]}
    return {
        "visible": visible, "partial": partial,
        "visible_text": [i for i in visible if types[i] in TEXT_TYPES],
        "label_any": int(bool(visible)),
        "label_text": int(any(types[i] in TEXT_TYPES for i in visible)),
        "ambiguous": bool(partial),
    }


def build(samples_dir: Path = SAMPLES_DIR, out_dir: Path = FRAMES_DIR, fps: float = FPS,
          long_side: int = 640) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sdir in sorted(Path(samples_dir).glob("s*")):
        if not (sdir / "annotations.json").exists():
            continue
        ann = annotations.load(sdir)
        key = source_key(ann)
        frames_dir = out_dir / "img" / ann["sample_id"]
        if _stale(frames_dir):
            _drop(out_dir, ann["sample_id"])
        frames = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
        if not frames:
            frames = extract(sdir / ann["video"]["file"], frames_dir, fps, long_side)
        for i, f in enumerate(frames):
            t = round(i / fps, 3)
            if t >= ann["video"]["duration"]:
                break
            rows.append({"frame": str(f.relative_to(out_dir)).replace("\\", "/"),
                         "sample_id": ann["sample_id"], "t": t, "source": key,
                         "split": split_for(key), **label(ann, t)})
    index = out_dir / "index.jsonl"
    with open(index, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return index


def load_index(out_dir: Path = FRAMES_DIR) -> list[dict]:
    with open(Path(out_dir) / "index.jsonl", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]
