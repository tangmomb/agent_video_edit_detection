"""Génération d'un échantillon : clip source + overlays aléatoires + annotations."""

import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from . import annotations, content, render
from .compose import Overlay, compose, probe
from .config import SAMPLES_DIR
from .fonts import available_fonts
from .sources import default_photo_source, default_video_source

ELEMENT_TYPES = ["keyword", "question", "subtitle", "lower_third", "image", "shape"]

# (fill, stroke, box)
TEXT_STYLES = [
    ((255, 255, 255, 255), (0, 0, 0, 255), None),
    ((255, 221, 0, 255), (0, 0, 0, 255), None),
    ((0, 0, 0, 255), None, (255, 214, 0, 255)),
    ((255, 255, 255, 255), None, (229, 37, 45, 255)),
    ((255, 255, 255, 255), None, (0, 0, 0, 170)),
    ((20, 20, 20, 255), None, (255, 255, 255, 240)),
    ((255, 255, 255, 255), None, (30, 110, 255, 255)),
]
ACCENTS = [(255, 214, 0, 255), (229, 37, 45, 255), (30, 110, 255, 255), (0, 200, 120, 255),
           (255, 120, 0, 255), (255, 255, 255, 255)]


@dataclass
class GenerationConfig:
    max_duration: float = 12.0
    min_duration: float = 4.0
    short_side: int = 720
    fps: int = 30
    n_elements: tuple[int, int] = (1, 6)
    empty_prob: float = 0.08  # échantillons sans aucun ajout (négatifs)
    lang: str = "fr"  # fr | en | mix
    type_weights: dict[str, float] = field(default_factory=lambda: {
        "keyword": 3, "question": 2, "subtitle": 2, "lower_third": 1, "image": 2, "shape": 1})


def _hex(c) -> str | None:
    return None if c is None else "#" + "".join(f"{v:02x}" for v in c)


def _overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class _Sampler:
    def __init__(self, rng: random.Random, cfg: GenerationConfig, W: int, H: int, D: float,
                 lang: str, topic: str, words: list[str], photo_source, photo_query: str):
        self.rng, self.cfg, self.W, self.H, self.D = rng, cfg, W, H, D
        self.lang, self.topic, self.words = lang, topic, words
        self.photo_source, self.photo_query = photo_source, photo_query
        self.short = min(W, H)
        self.margin = int(0.04 * self.short)
        self.fonts = available_fonts()
        self.placed: list[dict] = []  # éléments annotés
        self.overlays: list[tuple[Image.Image, tuple, dict]] = []

    # --- tirages -------------------------------------------------------------
    def timing(self, min_len=1.0, max_len=4.0) -> tuple[float, float]:
        length = self.rng.uniform(min(min_len, self.D), min(max_len, self.D))
        start = self.rng.uniform(0, self.D - length)
        return round(start, 2), round(start + length, 2)

    def animation(self, length: float, allow_slide: bool = True) -> dict:
        r = self.rng
        kinds = ["none", "fade"] + (["slide_left", "slide_right", "slide_up", "slide_down"] if allow_slide else [])
        weights = [3, 3] + ([1, 1, 1, 1] if allow_slide else [])
        kind = r.choices(kinds, weights)[0]
        budget = length / 2.5
        ain = {"type": kind, "duration": 0.0 if kind == "none" else round(min(r.uniform(0.15, 0.5), budget), 2)}
        if kind.startswith("slide"):
            d = r.uniform(0.1, 0.3)
            ain["offset"] = {"slide_left": [-d * self.W, 0], "slide_right": [d * self.W, 0],
                             "slide_up": [0, d * self.H], "slide_down": [0, -d * self.H]}[kind]
            ain["offset"] = [round(v, 1) for v in ain["offset"]]
            ain["fade"] = r.random() < 0.5
        out_kind = r.choice(["none", "fade"])
        aout = {"type": out_kind, "duration": 0.0 if out_kind == "none" else round(min(r.uniform(0.15, 0.4), budget), 2)}
        return {"in": ain, "out": aout}

    def font(self):
        return self.rng.choice(self.fonts)

    def text_style(self):
        return self.rng.choice(TEXT_STYLES)

    def photo(self) -> tuple[Image.Image, dict]:
        if self.photo_source is not None:
            try:
                return self.photo_source.pick(self.rng, self.photo_query)
            except Exception as e:
                print(f"photo indisponible, image synthétique utilisée : {e}")
        return render.synthetic_picture(self.rng), {"source": "synthetic"}

    # --- placement -----------------------------------------------------------
    def fit(self, make, max_h_ratio=0.5):
        """Appelle make(scale) en réduisant jusqu'à ce que l'élément tienne dans le cadre."""
        scale = 1.0
        for _ in range(6):
            img, box = make(scale)
            cw, ch = box[2] - box[0], box[3] - box[1]
            if cw <= self.W - 2 * self.margin and ch <= self.H * max_h_ratio:
                return img, box
            scale *= 0.82
        return None

    def position(self, kind: str, cw: int, ch: int, y_hint: int | None = None):
        r, m, W, H = self.rng, self.margin, self.W, self.H
        if kind == "subtitle":
            return (W - cw) // 2, min(y_hint, H - m - ch)
        if kind == "lower_third":
            return m + r.randint(0, int(W * 0.04)), min(int(H * r.uniform(0.62, 0.8)), H - m - ch)
        return r.randint(m, W - m - cw), r.randint(m, H - m - ch)

    def add(self, kind: str, rendered, t0: float, t1: float, anim: dict, content_value,
            text: str | None, style: dict, group: str | None = None, y_hint: int | None = None,
            tries: int = 25) -> bool:
        img, box = rendered
        cw, ch = box[2] - box[0], box[3] - box[1]
        for _ in range(tries if kind != "subtitle" else 1):
            x, y = self.position(kind, cw, ch, y_hint)
            bbox = [x, y, cw, ch]
            clash = any(
                _overlap(bbox, el["bbox"]) and t0 < el["t_end"] and el["t_start"] < t1
                for el in self.placed)
            if not clash:
                break
        else:
            return False
        el = {
            "id": f"el_{len(self.placed):02d}", "type": kind, "content": content_value, "text": text,
            "t_start": t0, "t_end": t1, "bbox": bbox, "animation": anim, "style": style,
        }
        if group:
            el["group"] = group
        self.placed.append(el)
        self.overlays.append((img, (x - box[0], y - box[1]), el))
        return True

    # --- types d'éléments ----------------------------------------------------
    def text_element(self, kind: str, text: str, size_range, width_ratio, t0, t1, anim,
                     style=None, group=None, y_hint=None, font=None, size=None) -> bool:
        r = self.rng
        fill, stroke, box = style or self.text_style()
        font = font or self.font()
        size = size or self.short * r.uniform(*size_range)
        max_w = self.W * r.uniform(*width_ratio) if width_ratio else None
        shadow = r.random() < 0.5

        def make(scale):
            s = int(size * scale)
            return render.render_text(
                text, font, s, fill,
                stroke_width=max(2, int(s * 0.08)) if stroke else 0, stroke_fill=stroke,
                box=box, box_radius=r.choice([0, int(s * 0.25)]) if box else 0,
                padding=int(s * 0.35), max_width=max_w * scale if max_w else None, shadow=shadow)

        rendered = self.fit(make)
        if not rendered:
            return False
        st = {"font": font.stem, "size": int(size), "fill": _hex(fill), "stroke": _hex(stroke), "box": _hex(box)}
        return self.add(kind, rendered, t0, t1, anim, text, text, st, group, y_hint)

    def keyword(self):
        t0, t1 = self.timing(0.8, 3.0)
        txt = content.keyword(self.rng, self.lang, self.words + self.topic.split())
        self.text_element("keyword", txt, (0.07, 0.13), None, t0, t1, self.animation(t1 - t0))

    def question(self):
        t0, t1 = self.timing(1.5, 4.5)
        txt = content.question(self.rng, self.lang, self.topic)
        self.text_element("question", txt, (0.045, 0.07), (0.55, 0.85), t0, t1, self.animation(t1 - t0))

    def subtitle(self, group: str):
        r = self.rng
        chunks = content.subtitle_chunks(r, self.lang, self.topic, n=r.randint(2, 6))
        style = r.choice([TEXT_STYLES[0], TEXT_STYLES[1], TEXT_STYLES[4]])
        font = self.font()
        size = self.short * r.uniform(0.045, 0.06)
        y_hint = int(self.H * r.uniform(0.72, 0.85))
        t = round(r.uniform(0, self.D * 0.5), 2)
        for chunk in chunks:
            dur = round(r.uniform(0.7, 1.6), 2)
            if t + dur > self.D:
                break
            anim = {"in": {"type": "none", "duration": 0.0}, "out": {"type": "none", "duration": 0.0}}
            self.text_element("subtitle", chunk, (0.045, 0.06), (0.85, 0.85), t, round(t + dur, 2),
                              anim, style=style, group=group, y_hint=y_hint, font=font, size=size)
            t = round(t + dur, 2)

    def lower_third(self):
        r = self.rng
        t0, t1 = self.timing(2.0, 5.0)
        title, sub = content.lower_third(r, self.lang, self.topic)
        bg = r.choice([(0, 0, 0, 200), (255, 255, 255, 235), (20, 30, 60, 230)])
        fg = (20, 20, 20, 255) if bg[0] > 128 else (255, 255, 255, 255)
        accent = r.choice(ACCENTS[:5])
        f1, f2 = self.font(), self.font()
        size = self.short * r.uniform(0.04, 0.055)
        rendered = self.fit(lambda s: render.render_lower_third(title, sub, f1, f2, int(size * s), bg, fg, accent))
        if rendered:
            st = {"font": f1.stem, "size": int(size), "bg": _hex(bg), "fg": _hex(fg), "accent": _hex(accent)}
            self.add("lower_third", rendered, t0, t1, self.animation(t1 - t0),
                     {"title": title, "subtitle": sub}, f"{title}\n{sub}", st)

    def image(self):
        r = self.rng
        t0, t1 = self.timing(1.0, 4.0)
        pic, meta = self.photo()
        width = int(self.W * r.uniform(0.2, 0.4) if self.W > self.H else self.W * r.uniform(0.35, 0.6))
        border = r.choice([0, 0, int(self.short * 0.008), int(self.short * 0.015)])
        radius = r.choice([0, int(self.short * 0.02), int(self.short * 0.04)])
        rendered = self.fit(lambda s: render.render_picture(pic, int(width * s), border=border, radius=radius,
                                                            shadow=r.random() < 0.6), max_h_ratio=0.6)
        if rendered:
            st = {"border": border, "radius": radius}
            self.add("image", rendered, t0, t1, self.animation(t1 - t0), meta, None, st)

    def shape(self):
        r = self.rng
        t0, t1 = self.timing(0.8, 3.0)
        kind = r.choice(["arrow", "circle", "star", "underline"])
        color = r.choice(ACCENTS)
        size = int(self.short * r.uniform(0.12, 0.25))
        angle = r.uniform(0, 360) if kind == "arrow" else 0.0
        rendered = self.fit(lambda s: render.render_shape(kind, int(size * s), color, max(3, int(size * s * 0.06)), angle))
        if rendered:
            st = {"color": _hex(color), "angle": round(angle, 1)}
            self.add("shape", rendered, t0, t1, self.animation(t1 - t0, allow_slide=False), {"shape": kind}, None, st)


def generate_sample(seed: int, out_root: Path = SAMPLES_DIR, config: GenerationConfig | None = None,
                    query: str | None = None, orientation: str | None = None,
                    video_source=None, photo_source=None, local_video: Path | None = None,
                    topic: str | None = None, accept_source=None) -> Path:
    """Génère data/samples/s<seed>/{video.mp4, annotations.json} et renvoie le dossier.

    accept_source(source) -> bool restreint les vidéos sources possibles (ex. : celles d'un split).
    """
    cfg = config or GenerationConfig()
    rng = random.Random(seed)

    if local_video:
        src_path = Path(local_video)
        topic_fr = topic or "la cuisine"
        photo_query = content.TOPIC_EN.get(topic_fr, "food")
        words: list[str] = []
        source = {"kind": "local", "path": str(src_path)}
    else:
        video_source = video_source or default_video_source()
        if query:
            topic_fr = topic or next((fr for en, fr in content.QUERY_POOL if en == query), query)
        else:
            query, topic_fr = rng.choice(content.QUERY_POOL)
        orientation = orientation or rng.choice(["portrait", "landscape"])
        extra = {"accept": accept_source} if accept_source else {}
        pick = video_source.pick(rng, query, orientation, cfg.min_duration, cfg.short_side, **extra)
        src_path, source, words = pick.path, pick.source, pick.words
        photo_query = query

    meta = probe(src_path)
    portrait = meta["height"] > meta["width"]
    short = cfg.short_side
    long_ = int(round(short * 16 / 9 / 2)) * 2
    W, H = (short, long_) if portrait else (long_, short)
    D = round(min(meta["duration"] - 0.1, cfg.max_duration), 2)
    ss = round(rng.uniform(0, max(0.0, meta["duration"] - D - 0.1)), 2)

    lang = content.pick_lang(rng, cfg.lang)
    topic_txt = content.topic_for(lang, topic_fr)
    if photo_source is None:
        photo_source = default_photo_source()
    sampler = _Sampler(rng, cfg, W, H, D, lang, topic_txt, words, photo_source, photo_query)

    n = 0 if rng.random() < cfg.empty_prob else rng.randint(*cfg.n_elements)
    types = list(cfg.type_weights)
    used: set[str] = set()
    for _ in range(n):
        kind = rng.choices(types, [cfg.type_weights[t] for t in types])[0]
        if kind in ("subtitle", "lower_third"):  # au plus un par vidéo
            if kind in used:
                kind = "keyword"
            used.add(kind)
        if kind == "subtitle":
            sampler.subtitle(group="sub_1")
        else:
            getattr(sampler, kind)()

    sample_id = f"s{seed:06d}"
    out_dir = Path(out_root) / sample_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        overlays = []
        for img, (px, py), el in sampler.overlays:
            png = Path(tmp) / f"{el['id']}.png"
            img.save(png)
            overlays.append(Overlay(png, px, py, el["t_start"], el["t_end"],
                                    el["animation"]["in"], el["animation"]["out"]))
        compose(src_path, out_dir / "video.mp4", overlays, W, H, cfg.fps, ss, D)

    elements = sorted(sampler.placed, key=lambda e: (e["t_start"], e["id"]))
    ann = {
        "schema_version": annotations.SCHEMA_VERSION,
        "sample_id": sample_id, "seed": seed, "lang": lang, "topic": topic_txt,
        "source": source,
        "video": {"file": "video.mp4", "width": W, "height": H, "fps": cfg.fps,
                  "duration": D, "source_offset": ss},
        "elements": elements,
    }
    annotations.save(out_dir, ann)
    return out_dir
