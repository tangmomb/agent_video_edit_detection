"""Planche de contrôle : quelques images d'une vidéo avec l'étiquette des annotateurs
(vert : ajout, gris : image brute, orange : désaccord, blanc : pas encore annotée)."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import FRAMES_DIR

COLORS = {1: "#34c759", 0: "#8e8e93", None: "#ff9f0a"}


def contact_sheet(sample_dir: str | Path, n: int = 12, cols: int = 4) -> Path:
    sample_dir = Path(sample_dir)
    frames = sorted((FRAMES_DIR / "img" / sample_dir.name).glob("*.jpg"))
    if not frames:
        raise ValueError(f"aucune image extraite pour {sample_dir.name} : lancer d'abord "
                         "python -m model_training_pipeline frames")
    votes_path = sample_dir / "votes.json"
    votes = json.loads(votes_path.read_text(encoding="utf-8"))["frames"] if votes_path.exists() else {}
    picked = [frames[round(i * (len(frames) - 1) / max(1, n - 1))] for i in range(min(n, len(frames)))]
    tiles = []
    for f in picked:
        img = Image.open(f).convert("RGB")
        font = ImageFont.load_default(size=max(14, max(img.size) // 30))
        v = votes.get(f.name)
        color = COLORS[v["label"]] if v else "white"
        text = (f"{int(f.stem) - 1}s " + ("pas annotée" if not v else "ajout" if v["label"] == 1
                                          else "brute" if v["label"] == 0 else "désaccord"))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width - 1, img.height - 1], outline=color, width=max(4, img.width // 80))
        d.text((10, 10), text, fill=color, font=font, stroke_width=2, stroke_fill="black")
        tiles.append(img)
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "black")
    for i, tile in enumerate(tiles):
        sheet.paste(tile.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    out = sample_dir / "preview.jpg"
    sheet.save(out, quality=88)
    return out
