"""Planche de contrôle : quelques frames de la vidéo avec les bbox annotées dessinées."""

from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from . import annotations

COLORS = {"keyword": "#ff3b30", "question": "#34c759", "subtitle": "#0a84ff",
          "lower_third": "#ff9f0a", "image": "#bf5af2", "shape": "#ffd60a"}


def _frames_at(video: Path, times: list[float], fps: float, size: tuple[int, int]) -> list[Image.Image]:
    wanted = sorted(set(int(round(t * fps)) for t in times))
    out: dict[int, Image.Image] = {}
    gen = imageio_ffmpeg.read_frames(str(video))
    next(gen)
    for i, raw in enumerate(gen):
        if i in wanted:
            out[i] = Image.frombytes("RGB", size, raw)
        if i >= wanted[-1]:
            break
    gen.close()
    return [out[int(round(t * fps))] for t in times if int(round(t * fps)) in out]


def contact_sheet(sample_dir: str | Path, n: int = 8, cols: int = 4) -> Path:
    sample_dir = Path(sample_dir)
    ann = annotations.load(sample_dir)
    v = ann["video"]
    D = v["duration"]
    # Instants : milieux d'éléments en priorité, complétés par une grille régulière.
    times = [round((e["t_start"] + e["t_end"]) / 2, 2) for e in ann["elements"]][:n]
    times += [round(D * (i + 0.5) / n, 2) for i in range(n - len(times))]
    times = sorted(times)
    frames = _frames_at(sample_dir / v["file"], times, v["fps"], (v["width"], v["height"]))

    big = max(v["width"], v["height"])
    font = ImageFont.load_default(size=max(14, big // 40))
    line_w = max(3, big // 200)
    tiles = []
    for t, img in zip(times, frames):
        d = ImageDraw.Draw(img)
        for el in annotations.visible_elements(ann, t):
            x, y, w, h = annotations.bbox_at(el, t)
            c = COLORS.get(el["type"], "white")
            d.rectangle([x, y, x + w, y + h], outline=c, width=line_w)
            d.text((x + 4, y - font.size - 4), f"{el['id']} {el['type']}", fill=c, font=font,
                   stroke_width=2, stroke_fill="black")
        d.text((10, 10), f"t={t:.2f}s", fill="white", font=font, stroke_width=2, stroke_fill="black")
        tiles.append(img)

    scale = 360 / max(v["width"], v["height"])
    tw, th = int(v["width"] * scale), int(v["height"] * scale)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "black")
    for i, tile in enumerate(tiles):
        sheet.paste(tile.resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
    out = sample_dir / "preview.jpg"
    sheet.save(out, quality=88)
    return out
