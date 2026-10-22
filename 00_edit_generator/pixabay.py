"""Client Pixabay (vidéos). Quota : 100 requêtes/60 s, cache 24 h obligatoire."""

import os

import requests

from .config import CACHE_DIR
from .http_cache import cached_json, download
from .ratelimit import QuotaExceeded, RateLimiter

API = "https://pixabay.com/api/videos/"
# Quota : 100 requêtes / 60 s. On attend jusqu'à 90 s plutôt que d'échouer.
LIMITER = RateLimiter("pixabay", max_calls=100, period_s=60, max_wait_s=90)
RETRIES = 3


class PixabayError(RuntimeError):
    pass


class PixabayClient:
    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.getenv("PIXABAY_API_KEY", "").strip()
        if not self.key:
            raise PixabayError("PIXABAY_API_KEY manquante.")
        self.session = requests.Session()

    def search_videos(self, query: str, per_page: int = 20, page: int = 1) -> list[dict]:
        params = {"q": query, "per_page": per_page, "page": page, "video_type": "film", "safesearch": "true"}

        def fetch():
            for _ in range(RETRIES):
                LIMITER.acquire()  # lève QuotaExceeded si l'attente serait trop longue
                r = self.session.get(API, params={**params, "key": self.key}, timeout=30)
                LIMITER.update(r.headers, limited=r.status_code == 429)
                if r.status_code == 429:
                    continue  # acquire() attendra la remise à zéro annoncée
                if not r.ok:
                    raise PixabayError(f"Pixabay {r.status_code}: {r.text[:200]}")
                return r.json()
            raise QuotaExceeded("quota Pixabay atteint (100 req/min) malgré les attentes")

        return cached_json("pixabay", params, fetch).get("hits", [])

    @staticmethod
    def size(hit: dict) -> tuple[int, int]:
        v = hit["videos"].get("medium") or next(iter(hit["videos"].values()))
        return v["width"], v["height"]

    def download_video(self, hit: dict, target_short_side: int = 720):
        dest = CACHE_DIR / "videos" / f"pixabay_{hit['id']}.mp4"
        if dest.exists():
            return dest
        files = [v for v in hit["videos"].values() if v.get("url") and v.get("width")]
        if not files:
            raise PixabayError(f"Aucun fichier pour la vidéo {hit['id']}.")
        big_enough = [f for f in files if min(f["width"], f["height"]) >= target_short_side]
        chosen = (min(big_enough, key=lambda f: f["width"] * f["height"]) if big_enough
                  else max(files, key=lambda f: f["width"] * f["height"]))
        return download(chosen["url"], dest, self.session)
