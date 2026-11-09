"""Garde-fous des pages publiques qui reçoivent des vidéos (modèle, saut d'images).

Sans mot de passe, chacun peut tester, dans des limites qui bornent le coût (GPU Modal, CPU, disque) :
une vidéo de taille et de durée plafonnées, quelques envois par adresse IP et par jour, un plafond
global par jour, un seul envoi traité à la fois, et le ménage des envois de plus de KEEP_H heures.

L'adresse du visiteur est la dernière d'X-Forwarded-For, posée par Caddy (qui ignore celle qu'un
client fournirait). Sans cet en-tête, la requête ne passe pas par Caddy (développement local) : pas
de quota. Les compteurs du jour sont dans <jobs>/quota.json et survivent aux redémarrages.
"""

import json
import re
import shutil
import subprocess
import time
from http import HTTPStatus
from pathlib import Path
from threading import Lock

KEEP_H = 24


def duration_s(video: Path) -> float | None:
    """Durée en secondes, lue dans la sortie de ffmpeg -i ; None si illisible."""
    from video_source.config import ffmpeg_bin

    err = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(video)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None


class Rejected(Exception):
    def __init__(self, message: str, status: int = HTTPStatus.TOO_MANY_REQUESTS):
        super().__init__(message)
        self.status = status


class Quota:
    def __init__(self, jobs: Path, per_ip: int, per_day: int, max_mb: int, max_s: int):
        self.jobs, self.per_ip, self.per_day = jobs, per_ip, per_day
        self.max_mb, self.max_s = max_mb, max_s
        self._busy, self._state = Lock(), Lock()

    @staticmethod
    def client(handler) -> str | None:
        """Adresse du visiteur derrière Caddy, None en local."""
        fwd = handler.headers.get("X-Forwarded-For")
        return fwd.split(",")[-1].strip() if fwd else None

    def limits(self) -> dict:
        return {"max_mb": self.max_mb, "max_min": round(self.max_s / 60, 1), "par_ip_jour": self.per_ip}

    # --- compteurs du jour -------------------------------------------------------------------------
    def _load(self) -> dict:
        path, today = self.jobs / "quota.json", time.strftime("%Y-%m-%d")
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        return state if state.get("day") == today else {"day": today, "total": 0, "ip": {}}

    def _save(self, state: dict) -> None:
        self.jobs.mkdir(parents=True, exist_ok=True)
        (self.jobs / "quota.json").write_text(json.dumps(state), encoding="utf-8")

    def _count(self, ip: str, delta: int) -> None:
        with self._state:
            s = self._load()
            s["total"] = max(0, s["total"] + delta)
            s["ip"][ip] = max(0, s["ip"].get(ip, 0) + delta)
            self._save(s)

    # --- admission d'un envoi ----------------------------------------------------------------------
    def admit(self, handler) -> str | None:
        """Accepte l'envoi ou lève Rejected ; rend l'IP comptée (None en local).
        En cas de succès, le verrou « un envoi à la fois » est pris : appeler release() ensuite."""
        ip = self.client(handler)
        if ip is None:
            if not self._busy.acquire(blocking=False):
                raise Rejected("Une autre vidéo est en cours de traitement, réessayez dans une minute.",
                               HTTPStatus.SERVICE_UNAVAILABLE)
            return None
        size = int(handler.headers.get("Content-Length") or 0)
        if size > self.max_mb * 1024 * 1024:
            raise Rejected(f"Vidéo trop lourde : {self.max_mb} Mo au plus.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        with self._state:
            s = self._load()
            if s["ip"].get(ip, 0) >= self.per_ip:
                raise Rejected(f"Limite atteinte : {self.per_ip} vidéos par jour. Revenez demain.")
            if s["total"] >= self.per_day:
                raise Rejected("Limite du jour atteinte pour tout le monde. Revenez demain.")
            if not self._busy.acquire(blocking=False):
                raise Rejected("Une autre vidéo est en cours de traitement, réessayez dans une minute.",
                               HTTPStatus.SERVICE_UNAVAILABLE)
            s["total"] += 1
            s["ip"][ip] = s["ip"].get(ip, 0) + 1
            self._save(s)
        return ip

    def check_video(self, ip: str | None, video: Path) -> None:
        """Après réception : refuse une vidéo trop longue ou illisible, sans la décompter du quota."""
        if ip is None:
            return
        d = duration_s(video)
        if d is None or d > self.max_s:
            self._count(ip, -1)
            raise Rejected("Fichier vidéo illisible." if d is None else
                           f"Vidéo trop longue : {self.max_s // 60} min au plus.", HTTPStatus.BAD_REQUEST)

    def refund(self, ip: str | None) -> None:
        if ip is not None:
            self._count(ip, -1)

    def release(self) -> None:
        self._busy.release()

    def cleanup(self) -> list[str]:
        """Efface les envois de plus de KEEP_H heures ; rend leurs noms."""
        if not self.jobs.exists():
            return []
        limit, gone = time.time() - KEEP_H * 3600, []
        for d in self.jobs.iterdir():
            if d.is_dir() and d.stat().st_mtime < limit:
                shutil.rmtree(d, ignore_errors=True)
                gone.append(d.name)
        return gone
