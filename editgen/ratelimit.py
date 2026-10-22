"""Respect des quotas des API d'images : fenêtre glissante persistée sur disque (les quotas
courent d'un processus à l'autre), attente quand la fenêtre est pleine, et prise en compte des
en-têtes X-RateLimit-* renvoyés par les API."""

import json
import time

from .config import CACHE_DIR


class QuotaExceeded(RuntimeError):
    """Quota épuisé et attente plus longue que ce que l'appelant accepte."""


class RateLimiter:
    def __init__(self, name: str, max_calls: int, period_s: float, max_wait_s: float, margin: int = 2):
        self.name = name
        self.max_calls = max_calls - margin  # marge pour les appels faits hors de ce processus
        self.period_s = period_s
        self.max_wait_s = max_wait_s
        self.path = CACHE_DIR / "api" / f"ratelimit_{name}.json"
        self.calls: list[float] = []
        self.blocked_until = 0.0
        self.remaining: int | None = None  # dernière valeur annoncée par l'API
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
                self.calls = state.get("calls", [])
                self.blocked_until = state.get("blocked_until", 0.0)
                self.remaining = state.get("remaining")
            except (ValueError, OSError):
                pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"calls": self.calls, "blocked_until": self.blocked_until,
                                         "remaining": self.remaining}), encoding="utf-8")

    def _prune(self, now: float) -> None:
        self.calls = [t for t in self.calls if now - t < self.period_s]

    def wait_time(self) -> float:
        now = time.time()
        self._prune(now)
        wait = max(0.0, self.blocked_until - now)
        if len(self.calls) >= self.max_calls:
            wait = max(wait, self.calls[0] + self.period_s - now)
        return wait

    def acquire(self) -> None:
        """Réserve un appel ; attend si nécessaire, ou lève QuotaExceeded si l'attente dépasse max_wait_s."""
        wait = self.wait_time()
        if wait > self.max_wait_s:
            raise QuotaExceeded(f"quota {self.name} épuisé, remise à zéro dans {wait:.0f} s")
        if wait > 0:
            print(f"[{self.name}] quota : attente de {wait:.0f} s")
            time.sleep(wait)
        self.calls.append(time.time())
        self._save()

    def update(self, headers, limited: bool = False, default_reset_s: float | None = None) -> None:
        """Met à jour l'état depuis la réponse : en-têtes X-RateLimit-* et/ou réponse « quota atteint »."""
        h = {k.lower(): v for k, v in headers.items()}
        if h.get("x-ratelimit-remaining", "").isdigit():
            self.remaining = int(h["x-ratelimit-remaining"])
        reset = h.get("retry-after") or h.get("x-ratelimit-reset")
        if limited or self.remaining == 0:
            reset_s = float(reset) if reset and reset.replace(".", "", 1).isdigit() else (
                default_reset_s if default_reset_s is not None else self.period_s)
            self.blocked_until = max(self.blocked_until, time.time() + reset_s)
        self._save()

    def status(self) -> dict:
        now = time.time()
        self._prune(now)
        return {"appels_dans_la_fenetre": len(self.calls), "max_par_fenetre": self.max_calls,
                "fenetre_s": self.period_s, "restant_annonce_par_api": self.remaining,
                "attente_avant_prochain_appel_s": round(self.wait_time())}


def api_quotas() -> dict:
    """État des quotas des API d'images (pour l'agent), sans déclencher d'appel."""
    from . import pixabay, unsplash
    return {"pixabay": pixabay.LIMITER.status(), "unsplash": unsplash.LIMITER.status()}
