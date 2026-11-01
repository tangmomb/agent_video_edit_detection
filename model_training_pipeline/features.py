"""Embeddings figés des frames, cache par échantillon
(data/videos_samples_frames/feat/<extracteur>__<vue>/<sample_id>.npz : ids, numéros des images
calculées (frame i -> i - 1), et emb, leurs embeddings ; seules les images demandées sont calculées).

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
    """Matrice (len(rows), dim) alignée sur rows. Cache : un fichier par vidéo avec les seules images
    déjà calculées (ids, emb) ; les images demandées et absentes du cache sont calculées et
    ajoutées (en pratique les images étiquetées, sauf pour le saut d'images, qui demande toute la
    vidéo en clip_b32:full)."""
    if extractor not in EXTRACTORS or view not in VIEWS:
        raise ValueError(f"embedding inconnu : {extractor}:{view} ; extracteurs {', '.join(EXTRACTORS)}, "
                         f"vues {', '.join(VIEWS)} (grid2 couvre les coins)")
    cdir = _cache_dir(frames_dir, extractor, view)
    cdir.mkdir(parents=True, exist_ok=True)
    by_sample: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_sample[r["sample_id"]].append(i)

    t0, n_frames, out = time.time(), 0, None
    for s, idxs in by_sample.items():
        path = cdir / f"{s}.npz"
        ids, emb = np.zeros(0, dtype=np.int32), None
        if path.exists():
            with np.load(path) as z:  # fermé tout de suite : Windows refuse de remplacer un fichier ouvert
                ids, emb = z["ids"], z["emb"]
        frame_ids = [int(Path(rows[i]["frame"]).stem) - 1 for i in idxs]
        todo = sorted(set(frame_ids) - set(ids.tolist()))
        if todo:
            frames = sorted((Path(frames_dir) / "img" / s).glob("*.jpg"))
            with torch.inference_mode():
                new = embed_images([frames[k] for k in todo], extractor, view, batch_size)
            ids = np.concatenate([ids, np.array(todo, dtype=np.int32)])
            emb = new.astype(np.float32) if emb is None else np.concatenate([emb, new.astype(np.float32)])
            order = np.argsort(ids)
            ids, emb = ids[order], emb[order]
            # fichier neuf puis remplacement : jamais de cache à moitié écrit, et un cache lié
            # (hard link d'un banc) n'est pas modifié à la source
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, ids=ids, emb=emb)
            tmp.replace(path)
            n_frames += len(todo)
        if out is None:
            out = np.zeros((len(rows), emb.shape[1]), dtype=np.float32)
        out[idxs] = emb[np.searchsorted(ids, frame_ids)]
    if n_frames:
        meta = {"ms_per_frame": round(1000 * (time.time() - t0) / n_frames, 2),
                "device": "cuda" if torch.cuda.is_available() else "cpu"}
        (cdir / "meta.json").write_text(json.dumps(meta))
    return out


def extraction_cost(extractor: str, view: str, frames_dir: Path = FRAMES_DIR) -> float | None:
    meta = _cache_dir(frames_dir, extractor, view) / "meta.json"
    return json.loads(meta.read_text())["ms_per_frame"] if meta.exists() else None


def precompute(extractors: list[str] | None = None, views: list[str] | None = None,
               frames_dir: Path = FRAMES_DIR) -> list[dict]:
    """Calcule d'avance les embeddings des images étiquetées (les seules qui servent à entraîner et
    mesurer) pour chaque extracteur et vue (par défaut tous ceux proposés à l'agent), afin que
    train_model n'ait plus qu'à les relire ; celui du saut d'images (SKIP_FEATURES) couvre toutes
    les images. Seules les images absentes du cache sont calculées."""
    from .frames import SKIP_FEATURES, load_index

    rows = load_index(frames_dir)
    labeled = [r for r in rows if r["label_any"] is not None]
    out = []
    for ext in extractors or list(EXTRACTORS):
        for view in views or list(VIEWS):
            want = rows if (ext, view) == SKIP_FEATURES else labeled
            print(f"[features] {ext}:{view} : {len(want)} images…", flush=True)
            t0 = time.time()
            get_features(want, ext, view, frames_dir)
            out.append({"embeddings": f"{ext}:{view}", "images": len(want), "duree_s": round(time.time() - t0, 1),
                        "ms_par_image": extraction_cost(ext, view, frames_dir)})
    return out
