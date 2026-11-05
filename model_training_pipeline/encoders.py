"""Extracteurs d'embeddings et vues d'une image : le calcul seul, sans cache ni dataset. Ce fichier
tourne tel quel sur le GPU distant (embed_modal.py) : un embedding s'y calcule comme ici.

Vues :
- full  : l'image entière, redimensionnée en carré (pas de recadrage central, qui
          couperait les sous-titres et les bandeaux en bord d'image) ;
- grid2 : l'image entière + ses 4 quadrants, embeddings concaténés (5x la dimension),
          pour mieux voir les petits overlays ;
- bands : l'image entière + son tiers bas + son tiers haut (pleine largeur, étirés en carré),
          concaténés (3x la dimension) : bandeaux de nom, sous-titres et titres y pèsent plus que
          dans l'image entière, où la scène domine (un plan de conférence brut pris pour monté) ;
- quads : les 4 quadrants seuls (4x), et tb : le tiers bas + le tiers haut seuls (2x). Sans
          l'image entière, pour la combiner avec full sans l'avoir deux fois (full + grid2 la
          doublait). Pas de cache propre : tranches de grid2 et bands, sans leur image entière.
          Proposées à l'agent à la place de grid2 et bands (03/11/2026), gardées pour les
          anciennes expériences.
"""

import numpy as np
import torch
from PIL import Image

EXTRACTORS = {
    "clip_b32": ("clip", "openai/clip-vit-base-patch32"),
    "clip_b16": ("clip", "openai/clip-vit-base-patch16"),
    "clip_l14": ("clip", "openai/clip-vit-large-patch14"),
    "dinov2_s": ("dino", "facebook/dinov2-small"),
    "dinov2_b": ("dino", "facebook/dinov2-base"),
}
VIEWS = {"full": 1, "grid2": 5, "bands": 3, "quads": 4, "tb": 2}  # vue -> nombre d'images par frame
AGENT_VIEWS = ("full", "quads", "tb")
DERIVED = {"quads": "grid2", "tb": "bands"}  # vue -> vue en cache dont elle est la fin
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
    crops = [] if view in DERIVED else [img]
    w, h = img.size
    if view in ("grid2", "quads"):
        crops += [img.crop((x, y, x + w // 2, y + h // 2)) for y in (0, h // 2) for x in (0, w // 2)]
    elif view in ("bands", "tb"):
        crops += [img.crop((0, h - h // 3, w, h)), img.crop((0, 0, w, h // 3))]
    return [c.resize((INPUT, INPUT), Image.BICUBIC) for c in crops]


def embed_images(paths: list, extractor: str, view: str = "full", batch_size: int = 64) -> np.ndarray:
    """Embeddings (len(paths), VIEWS[view] x dim) d'images quelconques (chemins ou fichiers ouverts),
    sans cache : utilisé par get_features et l'interface. À appeler sous torch.inference_mode()."""
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
