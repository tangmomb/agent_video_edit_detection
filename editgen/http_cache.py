"""Petit cache disque pour les réponses JSON des API (quotas serrés, et Pixabay
exige un cache de 24 h)."""

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import requests

from .config import CACHE_DIR


def cached_json(namespace: str, key: dict, fetch: Callable[[], dict], max_age: float = 24 * 3600) -> dict:
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE_DIR / "api" / namespace / f"{digest}.json"
    if path.exists() and time.time() - path.stat().st_mtime < max_age:
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def download(url: str, dest: Path, session: requests.Session | None = None) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with (session or requests).get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest
