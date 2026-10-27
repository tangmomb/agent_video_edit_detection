"""Embeddings figés des frames, cache par échantillon
(data/videos_samples_frames/feat/<extracteur>__<vue>/<sample_id>.npy).

Vues :
- full  : l'image entière, redimensionnée en carré (pas de recadrage central, qui
          couperait les sous-titres et les bandeaux en bord d'image) ;
- grid2 : l'image entière + ses 4 quadrants, embeddings concaténés (5x la dimension),
          pour mieux voir les petits overlays ;
- bands : l'image entière + son tiers bas + son tiers haut (pleine largeur, étirés en carré),
          concaténés (3x la dimension) : bandeaux de nom, sous-titres et titres y pèsent plus que
          dans l'image entière, où la scène domine (un plan de conférence brut pris pour monté).
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .frames import FRAMES_DIR

EXTRACTORS = {
    "clip_b32": ("clip", "openai/clip-vit-base-patch32"),
    "clip_b16": ("clip", "openai/clip-vit-base-patch16"),
    "clip_l14": ("clip", "openai/clip-vit-large-patch14"),
    "dinov2_s": ("dino", "facebook/dinov2-small"),
    "dinov2_b": ("dino", "facebook/dinov2-base"),
}
VIEWS = {"full": 1, "grid2": 5, "bands": 3}  # vue -> nombre d'images par frame
INPUT = 224

_encoders: dict = {}


def _encoder(name: str, device: str):
    if name in _encoders:
        return _encoders[name]
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    kind, repo = EXTRACTORS[name]
    if kind == "clip":
        model = CLIPModel.from_pretrained(repo).to(device).eval()
        proc = CLIPProcessor.from_pretrained(repo)

        def encode(images):
            inputs = proc(images=images, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs)
            if not isinstance(out, torch.Tensor):  # selon la version de transformers
                out = out.pooler_output
            return out
    else:
        model = AutoModel.from_pretrained(repo).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(repo)

        def encode(images):
            inputs = proc(images=images, return_tensors="pt", do_center_crop=False,
                          size={"height": INPUT, "width": INPUT}).to(device)
            h = model(**inputs).last_hidden_state
            # CLS + moyenne des patchs : le CLS seul rate souvent les petits overlays.
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
    _encoders[name] = encode
    return encode


def _views(img: Image.Image, view: str) -> list[Image.Image]:
    crops = [img]
    w, h = img.size
    if view == "grid2":
        crops += [img.crop((x, y, x + w // 2, y + h // 2)) for y in (0, h // 2) for x in (0, w // 2)]
    elif view == "bands":
        crops += [img.crop((0, h - h // 3, w, h)), img.crop((0, 0, w, h // 3))]
    return [c.resize((INPUT, INPUT), Image.BICUBIC) for c in crops]


def embed_images(paths: list[Path], extractor: str, view: str = "full", batch_size: int = 64) -> np.ndarray:
    """Embeddings (len(paths), VIEWS[view] x dim) d'images quelconques, sans cache : utilisé par
    get_features et l'interface. À appeler sous
    torch.inference_mode()."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode, n_views = _encoder(extractor, device), VIEWS[view]
    per_batch = max(1, batch_size // n_views)
    out = []
    with torch.autocast(device, enabled=device == "cuda"):
        for i in range(0, len(paths), per_batch):
            imgs = [v for f in paths[i:i + per_batch] for v in _views(Image.open(f).convert("RGB"), view)]
            emb = encode(imgs).float().cpu().numpy()
            out.append(emb.reshape(-1, n_views * emb.shape[1]))
    return np.concatenate(out)


def _cache_dir(frames_dir: Path, extractor: str, view: str) -> Path:
    return Path(frames_dir) / "feat" / f"{extractor}__{view}"


def get_features(rows: list[dict], extractor: str, view: str = "full",
                 frames_dir: Path = FRAMES_DIR, batch_size: int = 64) -> np.ndarray:
    """Matrice (len(rows), dim) alignée sur rows ; calcule seulement les échantillons manquants."""
    if extractor not in EXTRACTORS or view not in VIEWS:
        raise ValueError(f"extracteur/vue inconnus : {extractor}/{view}")
    cdir = _cache_dir(frames_dir, extractor, view)
    cdir.mkdir(parents=True, exist_ok=True)
    by_sample: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_sample[r["sample_id"]].append(i)

    missing = [s for s in by_sample if not (cdir / f"{s}.npy").exists()]
    if missing:
        t0, n_frames = time.time(), 0
        with torch.inference_mode():
            for s in missing:
                frames = sorted((Path(frames_dir) / "img" / s).glob("*.jpg"))
                np.save(cdir / f"{s}.npy", embed_images(frames, extractor, view, batch_size))
                n_frames += len(frames)
        if n_frames:
            meta = {"ms_per_frame": round(1000 * (time.time() - t0) / n_frames, 2),
                    "device": "cuda" if torch.cuda.is_available() else "cpu"}
            (cdir / "meta.json").write_text(json.dumps(meta))

    dim = None
    out = None
    for s, idxs in by_sample.items():
        arr = np.load(cdir / f"{s}.npy")
        if out is None:
            dim = arr.shape[1]
            out = np.zeros((len(rows), dim), dtype=np.float32)
        frame_ids = [int(Path(rows[i]["frame"]).stem) - 1 for i in idxs]
        out[idxs] = arr[frame_ids]
    return out


def extraction_cost(extractor: str, view: str, frames_dir: Path = FRAMES_DIR) -> float | None:
    meta = _cache_dir(frames_dir, extractor, view) / "meta.json"
    return json.loads(meta.read_text())["ms_per_frame"] if meta.exists() else None
