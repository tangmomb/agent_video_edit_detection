"""Écriture d'un fichier qui résiste à un arrêt brutal de la machine : on garde l'ancien contenu ou
le nouveau, jamais un fichier rempli de zéros (run 009 : son journal et exp_161 perdus)."""

import os
import time
from pathlib import Path


def write_atomic(path: Path, data: str | bytes) -> None:
    """Écrit data (texte en UTF-8) dans un fichier voisin, le force sur disque, puis remplace path."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data.encode("utf-8") if isinstance(data, str) else data)
            fh.flush()
            os.fsync(fh.fileno())
        # Sous Windows, un fichier ouvert en lecture ailleurs ne peut pas être remplacé : un lecteur
        # du journal (interface/run_chat, toutes les 5 s) faisait échouer l'outil en cours (run 008).
        for wait in (0.05, 0.1, 0.2, 0.5, 1.0, None):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if wait is None:
                    raise
                time.sleep(wait)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    sync_dir(path.parent)


def sync_dir(path: Path) -> None:
    """Force sur disque le contenu d'un dossier (nouveaux noms, remplacements). Rien à faire sous
    Windows, qui n'ouvre pas les dossiers."""
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
