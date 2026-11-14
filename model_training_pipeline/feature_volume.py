"""Copie des caches d'embeddings (<FRAMES_DIR>/feat/<extracteur>__<vue>/<vidéo>.npz) dans le volume
Modal agent-features, que lit l'entraînement sur GPU (train_modal.py).

Seuls les fichiers nouveaux ou modifiés (taille, date) depuis le dernier envoi partent : un
manifeste local (feat/.modal_volume.json) garde ce qui a été envoyé. Chaque dossier de caches a son
espace dans le volume (« main » pour celui du projet, un autre par dossier de banc, AGENT_LAB_DIR) :
un banc qui complète ses caches liés en dur ne remplace pas ceux du projet.
"""

import hashlib
import json
from pathlib import Path

from video_source import retry

from .frames import FRAMES_DIR

VOLUME = "agent-features"  # le même que train_modal.VOLUME (fichier copié seul dans l'image Modal)
MANIFEST = ".modal_volume.json"
DEFAULT_FEAT = FRAMES_DIR / "feat"


def namespace(feat_dir: Path) -> str:
    feat_dir = Path(feat_dir)
    if feat_dir.resolve() == DEFAULT_FEAT.resolve():
        return "main"
    return "lab-" + hashlib.sha1(str(feat_dir.resolve()).encode()).hexdigest()[:10]


def sync(feat_dir: Path, files: list[str], force: set[str] = frozenset(), say=print) -> int:
    """Envoie les fichiers (chemins relatifs à feat_dir) absents du manifeste ou changés depuis, et
    ceux de force (que le GPU n'a pas trouvés). Renvoie le nombre de fichiers envoyés."""
    import modal

    feat_dir = Path(feat_dir)
    man_path = feat_dir / MANIFEST
    try:
        man = json.loads(man_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        man = {}
    todo = []
    for rel in sorted(set(files)):
        st = (feat_dir / rel).stat()
        key = [st.st_size, st.st_mtime_ns]
        if rel in force or man.get(rel) != key:
            todo.append((rel, key))
    if not todo:
        return 0
    ns = namespace(feat_dir)
    mb = sum(k[0] for _, k in todo) / 1e6
    say(f"[volume] envoi de {len(todo)} fichiers d'embeddings ({mb:.0f} Mo) vers {VOLUME}/{ns}")
    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

    def send():
        with vol.batch_upload(force=True) as batch:
            for rel, _ in todo:
                batch.put_file(str(feat_dir / rel), f"/{ns}/{rel}")

    retry.forever(send, "volume d'embeddings")
    for rel, key in todo:
        man[rel] = key
    tmp = man_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(man), encoding="utf-8")
    tmp.replace(man_path)
    return len(todo)
