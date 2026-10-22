"""Rendu Pillow des éléments d'overlay en images RGBA.

Chaque fonction renvoie (image, content_box) où content_box = (x0, y0, x1, y1)
est la zone visible de l'élément dans l'image, ombre portée exclue.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

Color = tuple[int, ...]
Box = tuple[int, int, int, int]

MARGIN = 16  # place laissée autour pour l'ombre


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), max(8, int(size)))


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: float | None) -> list[str]:
    if not max_width:
        return [text]
    lines: list[str] = []
    cur = ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if not cur or font.getlength(cand) <= max_width:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _finish(img: Image.Image, shadow: bool) -> tuple[Image.Image, Box]:
    box = img.getchannel("A").getbbox() or (0, 0, 1, 1)
    if not shadow:
        return img, box
    alpha = img.getchannel("A").point(lambda a: a * 150 // 255)
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sh.putalpha(alpha)
    sh = sh.filter(ImageFilter.GaussianBlur(5))
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(sh, (4, 5))
    out.alpha_composite(img)
    return out, box


def render_text(text: str, font_path: Path, size: int, fill: Color,
                stroke_width: int = 0, stroke_fill: Color | None = None,
                box: Color | None = None, box_radius: int = 0, padding: int = 0,
                max_width: float | None = None, align: str = "center",
                shadow: bool = False, line_spacing: float = 1.12) -> tuple[Image.Image, Box]:
    font = _font(font_path, size)
    lines = _wrap(text, font, max_width)
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * line_spacing)
    widths = [font.getlength(l) for l in lines]
    sw = stroke_width
    text_w = int(math.ceil(max(widths))) + 2 * sw
    text_h = line_h * (len(lines) - 1) + ascent + descent + 2 * sw
    pad = padding if box else 0
    W = text_w + 2 * (pad + MARGIN)
    H = text_h + 2 * (pad + MARGIN)

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if box:
        d.rounded_rectangle([MARGIN, MARGIN, W - MARGIN - 1, H - MARGIN - 1],
                            radius=box_radius, fill=box)
    x0 = MARGIN + pad + sw
    y0 = MARGIN + pad + sw
    inner_w = text_w - 2 * sw
    for i, (line, lw) in enumerate(zip(lines, widths)):
        if align == "center":
            x = x0 + (inner_w - lw) / 2
        elif align == "right":
            x = x0 + inner_w - lw
        else:
            x = x0
        d.text((x, y0 + i * line_h), line, font=font, fill=fill,
               stroke_width=sw, stroke_fill=stroke_fill)
    return _finish(img, shadow)


def render_lower_third(title: str, subtitle: str, font_title: Path, font_sub: Path,
                       size: int, bg: Color, fg: Color, accent: Color,
                       shadow: bool = True) -> tuple[Image.Image, Box]:
    ft = _font(font_title, size)
    fs = _font(font_sub, int(size * 0.62))
    pad = int(size * 0.45)
    bar = max(4, int(size * 0.18))
    ta, td = ft.getmetrics()
    sa, sd = fs.getmetrics()
    gap = int(size * 0.15)
    inner_w = int(math.ceil(max(ft.getlength(title), fs.getlength(subtitle))))
    W = bar + inner_w + 2 * pad + 2 * MARGIN
    H = (ta + td) + gap + (sa + sd) + 2 * pad + 2 * MARGIN

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([MARGIN, MARGIN, W - MARGIN - 1, H - MARGIN - 1], fill=bg)
    d.rectangle([MARGIN, MARGIN, MARGIN + bar - 1, H - MARGIN - 1], fill=accent)
    tx = MARGIN + bar + pad
    d.text((tx, MARGIN + pad), title, font=ft, fill=fg)
    d.text((tx, MARGIN + pad + ta + td + gap), subtitle, font=fs, fill=accent)
    return _finish(img, shadow)


def render_picture(src: Image.Image, width: int, border: int = 0,
                   border_color: Color = (255, 255, 255, 255), radius: int = 0,
                   shadow: bool = True) -> tuple[Image.Image, Box]:
    src = src.convert("RGBA")
    h = max(1, round(src.height * width / src.width))
    pic = src.resize((width, h), Image.LANCZOS)
    W, H = width + 2 * (border + MARGIN), h + 2 * (border + MARGIN)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if border:
        d.rounded_rectangle([MARGIN, MARGIN, W - MARGIN - 1, H - MARGIN - 1],
                            radius=radius + border if radius else 0, fill=border_color)
    if radius:
        mask = Image.new("L", pic.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, pic.width - 1, pic.height - 1],
                                               radius=radius, fill=255)
        pic.putalpha(mask)
    img.alpha_composite(pic, (MARGIN + border, MARGIN + border))
    return _finish(img, shadow)


def render_shape(kind: str, size: int, color: Color, stroke: int,
                 angle: float = 0.0, shadow: bool = True) -> tuple[Image.Image, Box]:
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if kind == "circle":
        d.ellipse([stroke, stroke, s - stroke - 1, int(s * 0.7)], outline=color, width=stroke)
    elif kind == "arrow":
        t = max(stroke, s // 9)
        mid = s // 2
        d.rectangle([s * 0.08, mid - t / 2, s * 0.62, mid + t / 2], fill=color)
        d.polygon([(s * 0.55, mid - t * 2.2), (s * 0.95, mid), (s * 0.55, mid + t * 2.2)], fill=color)
    elif kind == "star":
        cx = cy = s / 2
        pts = []
        for i in range(10):
            r = s * 0.47 if i % 2 == 0 else s * 0.19
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        d.polygon(pts, fill=color)
    elif kind == "underline":
        h = max(stroke, s // 12)
        d.rounded_rectangle([0, s / 2 - h / 2, s - 1, s / 2 + h / 2], radius=h // 2, fill=color)
    else:
        raise ValueError(f"Forme inconnue : {kind}")
    if angle:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    canvas = Image.new("RGBA", (img.width + 2 * MARGIN, img.height + 2 * MARGIN), (0, 0, 0, 0))
    canvas.alpha_composite(img, (MARGIN, MARGIN))
    return _finish(canvas, shadow)


def synthetic_picture(rng, w: int = 640, h: int = 480) -> Image.Image:
    """Image de remplacement quand aucune photo n'est disponible."""
    c1 = tuple(rng.randint(0, 255) for _ in range(3))
    c2 = tuple(rng.randint(0, 255) for _ in range(3))
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        k = y / (h - 1)
        d.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * k) for a, b in zip(c1, c2)))
    for _ in range(rng.randint(2, 5)):
        x, y, r = rng.randint(0, w), rng.randint(0, h), rng.randint(20, h // 3)
        d.ellipse([x - r, y - r, x + r, y + r],
                  fill=tuple(rng.randint(0, 255) for _ in range(3)))
    return img
