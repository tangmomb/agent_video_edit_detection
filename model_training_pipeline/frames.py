"""Construit le dataset de frames étiquetées à partir des vraies vidéos de video_source.

Les images sont extraites à FPS par seconde, puis chacune est soumise aux trois LLM
d'annotators.ANNOTATORS (annotate, appels payants) ; les votes sont gardés dans votes.json à côté
de la vidéo. build() ne fait aucun appel payant : il relit les votes.

Chaque ligne de index.jsonl :
  frame, sample_id, t, source (chaîne YouTube : le split se fait par chaîne, voir lab.make_split),
  label_any    : 1 / 0 si les trois LLM sont d'accord (ajout de montage ou image brute),
                 None en cas de désaccord ou d'image pas encore annotée (jamais utilisée pour
                 entraîner ni mesurer, mais gardée pour la composante temporelle)
"""

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from video_source import annotators
from video_source.config import FRAMES_DIR, SAMPLES_DIR, ffmpeg_bin
from video_source.youtube import load_meta

FPS = 1.0  # frames extraites par seconde de vidéo
LONG_SIDE = 640
WORKERS = 8  # images annotées en parallèle


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
    """Supprime les frames d'un échantillon et leurs embeddings en cache (données dérivées)."""
    shutil.rmtree(out_dir / "img" / sample_id)
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


def load_votes(sdir: Path) -> dict:
    """votes.json : {"frames": {image: {"votes": {modèle: bool|None}, "label": 1|0|None}}, "cost_usd"}."""
    path = Path(sdir) / "votes.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"frames": {}, "cost_usd": 0.0}


def save_votes(sdir: Path, votes: dict) -> None:
    (Path(sdir) / "votes.json").write_text(json.dumps(votes, ensure_ascii=False), encoding="utf-8")


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


def pending(sdir: Path, out_dir: Path = FRAMES_DIR) -> list[Path]:
    """Images de la vidéo auxquelles il manque au moins un vote."""
    votes = load_votes(sdir)
    return [f for f in frames_of(sdir, out_dir) if missing(votes, f.name)]


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
                         "label_any": votes.get(f.name, {}).get("label")})
    index = out_dir / "index.jsonl"
    with open(index, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return index


def load_index(out_dir: Path = FRAMES_DIR) -> list[dict]:
    with open(Path(out_dir) / "index.jsonl", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]
