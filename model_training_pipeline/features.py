"""Embeddings figés des frames, cache par échantillon
(data/videos_samples_frames/feat/<extracteur>__<vue>/<sample_id>.npz : ids, numéros des images
calculées (frame i -> i - 1), et emb, leurs embeddings ; seules les images demandées sont calculées).
Extracteurs et vues : encoders.py. Calcul sur place, ou sur un GPU distant avec EMBED_GPU=modal
(embed_modal.py).
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from video_source import retry

from .encoders import AGENT_VIEWS, DERIVED, EXTRACTORS, VIEWS, _encoder, embed_images  # noqa: F401
from .frames import FRAMES_DIR

MODAL_CHUNK = 256  # images par appel au GPU distant (environ 8 Mo)


def embed_modal(paths: list[Path], extractor: str, view: str, wait_s: float | None = None) -> np.ndarray:
    """Embeddings calculés sur le GPU distant. Sans wait_s, une panne ou un manque de place met le
    calcul en pause (un run peut attendre) ; avec wait_s, tout morceau plus long lève une erreur, pour
    qui ne peut pas attendre (quelqu'un devant la page). L'appel est annulé : il ne s'exécutera pas plus tard."""
    import modal

    from .embed_modal import APP
    fn = modal.Function.from_name(APP, "embed")

    def call(jpegs: list[bytes]) -> bytes:
        if wait_s is None:
            return retry.forever(lambda: fn.remote(jpegs, extractor, view), "embeddings GPU")
        running = fn.spawn(jpegs, extractor, view)
        try:
            return running.get(timeout=wait_s)
        except modal.exception.TimeoutError:
            running.cancel()
            raise

    out = []
    for i in range(0, len(paths), MODAL_CHUNK):
        jpegs = [p.read_bytes() for p in paths[i:i + MODAL_CHUNK]]
        out.append(np.frombuffer(call(jpegs), dtype=np.float32).reshape(len(jpegs), -1))
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
                         f"vues {', '.join(AGENT_VIEWS)} (quads couvre les coins)")
    if view in DERIVED:
        parent = DERIVED[view]
        base = get_features(rows, extractor, parent, frames_dir, batch_size)
        return None if base is None else base[:, base.shape[1] // VIEWS[parent]:]
    cdir = _cache_dir(frames_dir, extractor, view)
    cdir.mkdir(parents=True, exist_ok=True)
    by_sample: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_sample[r["sample_id"]].append(i)

    remote = os.environ.get("EMBED_GPU") == "modal"
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
            if remote:
                new = embed_modal([frames[k] for k in todo], extractor, view)
            else:
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
    # Pas de mesure au GPU distant : ms_par_frame dit ce que coûte le modèle en production, sur place.
    if n_frames and not remote:
        meta = {"ms_per_frame": round(1000 * (time.time() - t0) / n_frames, 2),
                "device": "cuda" if torch.cuda.is_available() else "cpu"}
        (cdir / "meta.json").write_text(json.dumps(meta))
    return out


def extraction_cost(extractor: str, view: str, frames_dir: Path = FRAMES_DIR) -> float | None:
    if view in DERIVED:  # au prorata des images encodées
        parent = extraction_cost(extractor, DERIVED[view], frames_dir)
        return None if parent is None else parent * VIEWS[view] / VIEWS[DERIVED[view]]
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
        for view in views or list(AGENT_VIEWS):
            want = rows if (ext, view) == SKIP_FEATURES else labeled
            print(f"[features] {ext}:{view} : {len(want)} images…", flush=True)
            t0 = time.time()
            get_features(want, ext, view, frames_dir)
            out.append({"embeddings": f"{ext}:{view}", "images": len(want), "duree_s": round(time.time() - t0, 1),
                        "ms_par_image": extraction_cost(ext, view, frames_dir)})
    return out
