"""Client Unsplash (photos uniquement). Quota démo : 50 requêtes/heure."""

import os

import requests

from .config import CACHE_DIR
from .http_cache import cached_json, download
from .ratelimit import QuotaExceeded, RateLimiter

API = "https://api.unsplash.com"
# Quota démo : 50 requêtes / heure. Pas d'attente : à court de quota, le générateur
# réutilise les photos en cache ou des images synthétiques.
LIMITER = RateLimiter("unsplash", max_calls=50, period_s=3600, max_wait_s=0)


class UnsplashError(RuntimeError):
    pass


class UnsplashClient:
    def __init__(self, access_key: str | None = None):
        key = access_key or os.getenv("UNSPLASH_ACCESS_KEY", "").strip()
        if not key:
            raise UnsplashError("UNSPLASH_ACCESS_KEY manquante.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Client-ID {key}", "Accept-Version": "v1"})

    def _get(self, url: str, params: dict | None = None) -> dict:
        try:
            LIMITER.acquire()
        except QuotaExceeded as e:
            raise UnsplashError(str(e)) from e
        r = self.session.get(url, params=params, timeout=30)
        limited = r.status_code == 429 or (r.status_code == 403 and r.headers.get("X-Ratelimit-Remaining") == "0")
        LIMITER.update(r.headers, limited=limited)
        if limited:
            raise UnsplashError("Quota Unsplash atteint (50 req/h en mode démo).")
        if not r.ok:
            raise UnsplashError(f"Unsplash {r.status_code}: {r.text[:200]}")
        return r.json()

    def search_photos(self, query: str, per_page: int = 30, page: int = 1,
                      orientation: str | None = None) -> list[dict]:
        params = {"query": query, "per_page": per_page, "page": page, "content_filter": "high"}
        if orientation:
            params["orientation"] = orientation
        data = cached_json("unsplash", params, lambda: self._get(f"{API}/search/photos", params))
        return data.get("results", [])

    def download_photo(self, photo: dict, size: str = "regular"):
        dest = CACHE_DIR / "photos" / f"unsplash_{photo['id']}_{size}.jpg"
        if not dest.exists():
            # Les guidelines de l'API exigent de déclencher download_location.
            loc = photo.get("links", {}).get("download_location")
            if loc:
                self._get(loc)
            download(photo["urls"][size], dest)
        return dest
