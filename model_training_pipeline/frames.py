"""Construit le dataset de frames étiquetées à partir des vraies vidéos de video_source.

Les images sont extraites à FPS par seconde, puis soumises aux trois LLM d'annotators.ANNOTATORS
(annotate, appels payants) ; les votes sont gardés dans votes.json à côté de ses images
(img/<sample_id>/, voir votes_path). build() ne
fait aucun appel payant : il relit les votes.

Une image presque identique à la dernière image annotée de sa vidéo n'est pas annotée (to_annotate) :
à 1 image par seconde, 4 sur 10 répètent la précédente, et les payer n'apprend rien au modèle.

Chaque ligne de index.jsonl :
  frame, sample_id, t, source (chaîne YouTube : le split se fait par chaîne, voir lab.make_split),
  label_any    : 1 / 0 si les trois LLM sont d'accord (ajout de montage ou image brute),
                 None en cas de désaccord ou d'image pas encore annotée (jamais utilisée pour
                 entraîner ni mesurer)
  annotee      : au moins un vote reçu (False : image sautée comme quasi identique, ou en attente)
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from video_source import annotators
from video_source.config import FRAMES_DIR, SAMPLES_DIR, ffmpeg_bin
from video_source.youtube import load_meta

FPS = 1.0  # frames extraites par seconde de vidéo
LONG_SIDE = 640
WORKERS = 8  # images annotées en parallèle
# Cosinus CLIP (clip_b32, image entière) au-delà duquel une image est jugée identique à la dernière
# annotée de la vidéo et sautée. Mesuré avec skip_refs sur les 7 144 images du dataset (27/10/2026) :
# 0.87 saute 69 % des images, mais 20 % des 230 changements d'étiquette (apparition ou disparition
# d'un ajout) tombent sur une image sautée et 2,5 % des images sautées ont une autre étiquette que
# leur référence ; 0.97 n'en sautait que 24 %, pour 1 % des changements et 0,1 % des étiquettes.
# Choisi à l'œil dans utils/skip_tester : à 0.97, trop d'images quasi identiques partaient encore.
SKIP_SIMILAR = 0.87
SKIPPED_SHARE = 0.69  # part des images sautées à ce seuil : estimation du coût d'une vidéo
SKIP_FEATURES = ("clip_b32", "full")  # embeddings comparés : calculés pour toutes les images


def source_key(meta: dict) -> str:
    return f"youtube:{meta['channel_id']}"


# Version de l'extraction, notée dans img/<sample_id>/.extraction : des frames extraites par une
# autre version (et leurs embeddings) sont refaites au prochain build().
EXTRACTION = "fps-round-up"


def extract(video: Path, out_dir: Path, fps: float, long_side: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = f"scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'"
    # round=up : la frame i est celle de l'instant i/fps. Avec le défaut (round=near), ffmpeg prend
    # la frame ~0,5 s plus tard (constaté sur les vidéos générées, octobre 2026).
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vf", f"fps={fps}:round=up,{scale}", "-q:v", "3", str(out_dir / "%05d.jpg")]
    subprocess.run(cmd, check=True)
    (out_dir / ".extraction").write_text(EXTRACTION)
    return sorted(out_dir.glob("*.jpg"))


def _stale(frames_dir: Path) -> bool:
    marker = frames_dir / ".extraction"
    return frames_dir.exists() and (not marker.exists() or marker.read_text() != EXTRACTION)


def _drop(out_dir: Path, sample_id: str) -> None:
    """Supprime les frames d'un échantillon et leurs embeddings en cache (données dérivées), mais
    garde ses votes (votes.json, payés), rangés dans le même dossier."""
    for f in (out_dir / "img" / sample_id).iterdir():
        if f.name != "votes.json":
            f.unlink()
    for f in (out_dir / "feat").glob(f"*/{sample_id}.npy"):
        f.unlink()


def frames_of(sdir: Path, out_dir: Path = FRAMES_DIR, fps: float = FPS, long_side: int = LONG_SIDE) -> list[Path]:
    """Images extraites de la vidéo (extraites au besoin), sans celles qu'ffmpeg ajoute parfois
    après la fin de la vidéo (instant >= durée)."""
    meta = load_meta(sdir)
    frames_dir = Path(out_dir) / "img" / meta["sample_id"]
    if _stale(frames_dir):
        _drop(Path(out_dir), meta["sample_id"])
    frames = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    frames = frames or extract(sdir / meta["video"]["file"], frames_dir, fps, long_side)
    duration = meta["video"]["duration"]
    return [f for i, f in enumerate(frames) if not duration or i / fps < duration]


def votes_path(sdir: Path) -> Path:
    """votes.json d'une vidéo (dossier sdir de videos_samples) : avec ses images, dans le dossier
    videos_samples_frames/img/ voisin (vrai aussi pour la copie du dataset d'un banc)."""
    sdir = Path(sdir)
    return sdir.parent.parent / FRAMES_DIR.name / "img" / sdir.name / "votes.json"


def load_votes(sdir: Path) -> dict:
    """votes.json : {"frames": {image: {"votes": {modèle: bool|None}, "label": 1|0|None}}, "cost_usd"}."""
    path = votes_path(sdir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"frames": {}, "cost_usd": 0.0}


def save_votes(sdir: Path, votes: dict) -> None:
    path = votes_path(sdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(votes, ensure_ascii=False), encoding="utf-8")


def add_votes(votes: dict, frame: str, new: dict, cost_usd: float) -> None:
    """Ajoute les votes de certains modèles pour une image (appel direct ou résultat de batch) et
    recalcule son étiquette : elle n'existe qu'une fois tous les annotateurs d'accord."""
    entry = votes["frames"].setdefault(frame, {"votes": {}, "label": None})
    entry["votes"].update(new)
    entry["label"] = annotators.consensus(entry["votes"])
    votes["cost_usd"] += cost_usd


def missing(votes: dict, frame: str) -> list[str]:
    """Annotateurs qui n'ont pas encore voté sur l'image."""
    got = votes["frames"].get(frame, {}).get("votes", {})
    return [m for m in annotators.ANNOTATORS if m not in got]


def skip_refs(X, voted: list[bool] | None = None, threshold: float = SKIP_SIMILAR) -> list[int]:
    """Pour chaque image (embeddings CLIP X dans l'ordre de la vidéo), l'image gardée à laquelle elle
    a été comparée : elle-même si elle est gardée, sinon la dernière gardée avant elle, dont le
    cosinus dépassait threshold. Une image déjà votée est toujours gardée. Comparer à la dernière
    gardée, et non à la précédente, rattrape une lente dérive."""
    import numpy as np

    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    refs, ref = [], None
    for i, x in enumerate(X):
        if (voted and voted[i]) or ref is None or float(x @ X[ref]) <= threshold:
            ref = i
        refs.append(ref)
    return refs


def to_annotate(sdir: Path, out_dir: Path = FRAMES_DIR, votes: dict | None = None) -> list[Path]:
    """Images de la vidéo à annoter (skip_refs) : sans celles dont le cosinus CLIP avec la dernière
    image gardée dépasse SKIP_SIMILAR. Embeddings calculés au besoin (mis en cache, réutilisés à
    l'entraînement)."""
    from .features import get_features

    frames = frames_of(sdir, out_dir)
    if not frames:
        return []
    votes = votes if votes is not None else load_votes(sdir)
    sid = load_meta(sdir)["sample_id"]
    X = get_features([{"sample_id": sid, "frame": f.name} for f in frames], *SKIP_FEATURES, Path(out_dir))
    refs = skip_refs(X, [bool(votes["frames"].get(f.name, {}).get("votes")) for f in frames])
    return [f for i, f in enumerate(frames) if refs[i] == i]


def pending(sdir: Path, out_dir: Path = FRAMES_DIR) -> list[Path]:
    """Images à annoter (to_annotate) auxquelles il manque au moins un vote."""
    votes = load_votes(sdir)
    return [f for f in to_annotate(sdir, out_dir, votes) if missing(votes, f.name)]


def annotate(sdir: Path, max_usd: float | None = None, apis: dict | None = None,
             out_dir: Path = FRAMES_DIR) -> dict:
    """Fait voter les annotateurs sur les images pas encore annotées de la vidéo (appels payants,
    un par image et par modèle manquant). S'arrête avant de dépasser max_usd ; reprend là où il
    s'était arrêté. Renvoie le bilan."""
    sdir = Path(sdir)
    votes = load_votes(sdir)
    todo = pending(sdir, out_dir)
    apis = apis if apis is not None else {}
    spent, done, stopped = 0.0, 0, False
    with ThreadPoolExecutor(WORKERS) as pool:
        for i in range(0, len(todo), WORKERS):
            chunk = todo[i:i + WORKERS]
            # le prochain lot coûterait à peu près autant par image que les précédents
            if max_usd is not None and done and spent + spent / done * len(chunk) > max_usd:
                stopped = True
                break
            asked = pool.map(lambda f: annotators.vote(f, apis, models=missing(votes, f.name)), chunk)
            for f, v in zip(chunk, asked):
                add_votes(votes, f.name, v["votes"], v["cost_usd"])
                spent += v["cost_usd"]
            done += len(chunk)
            # écrit après chaque lot : une annotation interrompue reprend là où elle s'est arrêtée
            save_votes(sdir, votes)
    labels = [v["label"] for v in votes["frames"].values()]
    return {"sample_id": sdir.name, "images_annotees": len(labels), "cost_usd": round(spent, 4),
            "images_avec_accord": sum(lab is not None for lab in labels),
            "images_restantes": len(todo) - done, "arret_budget": stopped}


def build(samples_dir: Path = SAMPLES_DIR, out_dir: Path = FRAMES_DIR, fps: float = FPS,
          long_side: int = LONG_SIDE) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sdir in sorted(Path(samples_dir).glob("yt_*")):
        if not (sdir / "meta.json").exists():
            continue
        meta = load_meta(sdir)
        votes = load_votes(sdir)["frames"]
        for i, f in enumerate(frames_of(sdir, out_dir, fps, long_side)):
            t = round(i / fps, 3)
            rows.append({"frame": str(f.relative_to(out_dir)).replace("\\", "/"),
                         "sample_id": meta["sample_id"], "t": t, "source": source_key(meta),
                         "label_any": votes.get(f.name, {}).get("label"),
                         "annotee": bool(votes.get(f.name, {}).get("votes"))})
    index = out_dir / "index.jsonl"
    with open(index, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return index


def load_index(out_dir: Path = FRAMES_DIR) -> list[dict]:
    with open(Path(out_dir) / "index.jsonl", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]
