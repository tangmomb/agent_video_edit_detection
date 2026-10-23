"""Sources de médias interchangeables.

- Vidéos : Pixabay (PIXABAY_API_KEY), ou les rushes déjà téléchargés (CachedVideos).
- Photos (overlays « image ») : Unsplash (UNSPLASH_ACCESS_KEY), les photos déjà téléchargées
  (CachedPhotos), sinon images synthétiques.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from .config import CACHE_DIR


@dataclass
class VideoPick:
    path: Path
    source: dict  # métadonnées écrites dans annotations.json
    words: list[str]  # mots descriptifs (tags, slug) pour les mots-clés


class PixabayVideos:
    def __init__(self):
        from .pixabay import PixabayClient
        self.client = PixabayClient()

    def pick(self, rng: random.Random, query: str, orientation: str, min_duration: float,
             short_side: int, accept: Callable[[dict], bool] | None = None) -> VideoPick:
        """accept(source) filtre les vidéos candidates sur leurs métadonnées (kind, video_id)."""
        # Le portrait est rare sur Pixabay : en dernier recours on accepte l'autre orientation.
        # Avec un filtre accept, on parcourt plus de pages : il écarte une partie des résultats
        # (rushes déjà utilisés, autre split).
        extra = [(p, True) for p in range(2, 6)] if accept else []
        pages = [(rng.randint(1, 3), True), (1, True)] + extra + [(1, False)]
        for page, strict in pages:
            hits = [h for h in self.client.search_videos(query, page=page)
                    if h.get("duration", 0) >= min_duration
                    and (not strict or (self.client.size(h)[1] > self.client.size(h)[0]) == (orientation == "portrait"))
                    and (accept is None or accept({"kind": "pixabay", "video_id": h["id"]}))]
            if hits:
                h = rng.choice(hits)
                words = [w.strip() for w in h.get("tags", "").split(",") if w.strip()]
                return VideoPick(self.client.download_video(h, short_side),
                                 {"kind": "pixabay", "video_id": h["id"], "url": h.get("pageURL"),
                                  "author": h.get("user"), "query": query}, words)
        raise RuntimeError(f"Aucune vidéo Pixabay pour '{query}' ({orientation})"
                           + (" qui passe le filtre de source (rush déjà utilisé ou autre split) : "
                              "change de thème." if accept else "."))


class UnsplashPhotos:
    def __init__(self):
        from .unsplash import UnsplashClient
        self.client = UnsplashClient()

    def pick(self, rng: random.Random, query: str) -> tuple[Image.Image, dict]:
        from .unsplash import UnsplashError
        try:
            photos = self.client.search_photos(query, page=rng.randint(1, 3))
            p = rng.choice(photos)
            path = self.client.download_photo(p)
        except UnsplashError:
            # Quota démo (50 req/h) : on réutilise une photo déjà téléchargée.
            cached = sorted((CACHE_DIR / "photos").glob("unsplash_*.jpg"))
            if not cached:
                raise
            path = rng.choice(cached)
            return Image.open(path), {"source": "unsplash", "photo_id": path.stem.split("_")[1], "cached": True}
        img = Image.open(path)
        return img, {"source": "unsplash", "photo_id": p["id"],
                     "alt": p.get("alt_description") or p.get("description") or "",
                     "author": (p.get("user") or {}).get("name"), "url": (p.get("links") or {}).get("html")}


class CachedVideos:
    """Rushes déjà téléchargés (cache/videos/pixabay_<id>.mp4), sans appel d'API. La source garde
    l'id Pixabay : un rush tiré du cache compte comme déjà utilisé pour une recherche Pixabay."""

    def __init__(self, paths: list[Path] | None = None):
        self.paths = sorted(paths if paths is not None else (CACHE_DIR / "videos").glob("pixabay_*.mp4"))

    def pick(self, rng: random.Random, query: str, orientation: str, min_duration: float,
             short_side: int, accept: Callable[[dict], bool] | None = None) -> VideoPick:
        from .compose import probe
        hits = []
        for p in self.paths:
            source = {"kind": "pixabay", "video_id": int(p.stem.split("_")[1]), "cached": True}
            if accept is not None and not accept(source):
                continue
            meta = probe(p)
            if meta["duration"] >= min_duration and (meta["height"] > meta["width"]) == (orientation == "portrait"):
                hits.append((p, source))
        if not hits:
            raise RuntimeError(f"Aucun rush en cache ({orientation}) qui ne soit pas déjà utilisé.")
        p, source = rng.choice(hits)
        return VideoPick(p, source, [])


class CachedPhotos:
    """Photos déjà téléchargées (cache/photos), sans appel d'API."""

    def __init__(self):
        self.paths = sorted((CACHE_DIR / "photos").glob("unsplash_*.jpg"))

    def pick(self, rng: random.Random, query: str) -> tuple[Image.Image, dict]:
        p = rng.choice(self.paths)
        return Image.open(p), {"source": "unsplash", "photo_id": p.stem.split("_")[1], "cached": True}


def default_video_source():
    if os.getenv("PIXABAY_API_KEY"):
        return PixabayVideos()
    raise RuntimeError("Aucune source vidéo : renseigner PIXABAY_API_KEY dans .env, "
                       "ou utiliser --local-video.")


def default_photo_source():
    if os.getenv("UNSPLASH_ACCESS_KEY"):
        return UnsplashPhotos()
    return None
