"""Vraies vidéos YouTube : recherche et téléchargement avec yt-dlp (aucune clé d'API).

Chaque vidéo est rangée dans SAMPLES_DIR/yt_<id YouTube>/ : video.<ext> (image seule, MAX_RES au plus,
le son ne sert pas) et meta.json (titre, chaîne, url, requête, durée, dimensions). Une vidéo n'est
téléchargée qu'une fois ; la chaîne sert de groupe pour le split (même habillage d'une vidéo à
l'autre).
"""

import json
import os
import random
import shutil
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote_plus

from .config import SAMPLES_DIR, ffmpeg_bin

MAX_DURATION = 300  # s : au-delà, la vidéo coûterait trop à annoter (une image par seconde)
MIN_DURATION = 5
# Plus petit côté de la vidéo téléchargée. Les images sont réduites à 640 px (frames.LONG_SIDE) et
# le modèle les voit en 224 : 360p (640×360) suffit. Comparé sur une vidéo à sous-titres (26/10/2026) :
# bandeau et sous-titres nets comme en 720p, pour 57 % de volume en moins (proxy, temps).
MAX_RES = 360
SEARCH_FACTOR = 4  # candidats cherchés par vidéo voulue (déjà prises, trop longues, mauvaise orientation)
MIN_CANDIDATES = 25  # plancher : avec peu de vidéos voulues, 4 par vidéo ne suffit pas à trouver une bonne
# Filtre de la recherche YouTube « vidéos de moins de 4 minutes » : sans lui, la plupart des
# résultats dépassent MAX_DURATION et une requête courante paraît épuisée.
SHORT_FILTER = "EgQQARgB"
REQUEST_PAUSE = (2, 6)  # s : attente aléatoire entre deux lectures de page vidéo, contre le blocage YouTube
DOWNLOAD_PAUSE = (60, 180)  # s : attente aléatoire entre deux téléchargements, pour ménager YouTube
# Relais de tout le trafic YouTube (.env), pour une machine dont YouTube bloque l'IP : le VPS Contabo
# (« not a bot » dès la 1re vidéo, IPv4 comme IPv6) passe par un tunnel SSH vers le VPS Infomaniak,
# YT_PROXY=socks5://127.0.0.1:1080 (service yt-tunnel). Vide : connexion directe.
PROXY = os.getenv("YT_PROXY")


class NoVideoFound(Exception):
    pass


def sample_id(video_id: str) -> str:
    return f"yt_{video_id}"


def downloaded(samples_dir: Path = SAMPLES_DIR) -> set[str]:
    """Ids YouTube déjà téléchargés."""
    return {json.loads(p.read_text(encoding="utf-8"))["video_id"] for p in Path(samples_dir).glob("yt_*/meta.json")}


def _ydl(**opts):
    import yt_dlp
    proxy = {"proxy": PROXY} if PROXY else {}
    return yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noprogress": True, **proxy, **opts})


def _orientation(info: dict) -> str | None:
    w, h = info.get("width"), info.get("height")
    if not (w and h):
        return None
    return "portrait" if h > w else "landscape"


def hashtag(query: str) -> str:
    """« Recette pâtes ! » -> « recettepates » (les hashtags YouTube n'ont ni espaces ni accents)."""
    s = unicodedata.normalize("NFKD", query.lower())
    return "".join(c for c in s if c.isalnum() and not unicodedata.combining(c))


def _candidates(query: str, n: int, orientation: str | None) -> list[dict]:
    """Résultats bruts (sans détail) : la recherche YouTube ne renvoie presque jamais de Shorts, le
    portrait passe donc par la page Shorts du hashtag tiré de la requête."""
    count = max(n * SEARCH_FACTOR, MIN_CANDIDATES)
    with _ydl(extract_flat=True, playlistend=count) as ydl:
        if orientation == "portrait":
            try:
                url = f"https://www.youtube.com/hashtag/{hashtag(query)}/shorts"
                return list(ydl.extract_info(url, download=False)["entries"])
            except Exception:
                pass  # hashtag inconnu : recherche classique, filtrée sur les dimensions
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={SHORT_FILTER}"
        return [e for e in ydl.extract_info(url, download=False)["entries"] if e]


def search(query: str, n: int, orientation: str | None = None, exclude: set[str] | None = None,
           avoid_channels: set[str] | None = None) -> list[dict]:
    """Jusqu'à n vidéos (infos complètes yt-dlp) pour la requête, jamais téléchargées, de durée
    acceptable et, si demandé, de la bonne orientation. Le portrait vise les Shorts.
    avoid_channels : chaînes à écarter (complété au fil de la recherche : une vidéo par chaîne)."""
    exclude = (exclude or set()) | downloaded()
    out, looked = [], 0
    with _ydl() as ydl:
        for e in _candidates(query, n, orientation):
            if len(out) >= n:
                break
            d = e.get("duration")  # absente des pages Shorts : vérifiée sur l'info complète
            if e["id"] in exclude or e.get("live_status") not in (None, "not_live") \
                    or (d is not None and not MIN_DURATION <= d <= MAX_DURATION) \
                    or (avoid_channels is not None and e.get("channel_id") in avoid_channels):
                continue
            if looked:
                time.sleep(random.uniform(*REQUEST_PAUSE))
            looked += 1
            try:
                info = ydl.extract_info(e["url"], download=False)  # dimensions : une requête, sans télécharger
            except Exception:
                continue  # vidéo privée, restreinte, retirée…
            if not MIN_DURATION <= (info.get("duration") or 0) <= MAX_DURATION:
                continue
            if orientation and _orientation(info) != orientation:
                continue
            channel = info.get("channel_id") or info.get("uploader_id")
            if avoid_channels is not None:
                if channel in avoid_channels:
                    continue
                avoid_channels.add(channel)
            exclude.add(e["id"])
            out.append(info)
    return out


def download(info: dict, query: str | None, samples_dir: Path = SAMPLES_DIR) -> Path:
    """Télécharge la vidéo (image seule, MAX_RES au plus) et écrit meta.json ; renvoie son dossier."""
    out = Path(samples_dir) / sample_id(info["id"])
    out.mkdir(parents=True, exist_ok=True)
    # res = plus petit côté : 360 vaut pour le paysage (640×360) comme pour le portrait (360×640).
    try:
        with _ydl(format="bv*/b", format_sort=[f"res:{MAX_RES}", "ext:mp4"],
                  outtmpl=str(out / "video.%(ext)s"), ffmpeg_location=ffmpeg_bin()) as ydl:
            got = ydl.extract_info(info["webpage_url"], download=True)
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise
    video = next(out.glob("video.*"))
    meta = {"sample_id": out.name, "video_id": info["id"], "url": info["webpage_url"],
            "title": info.get("title"), "channel": info.get("channel"),
            "channel_id": info.get("channel_id") or info.get("uploader_id"), "query": query,
            "video": {"file": video.name, "width": got.get("width"), "height": got.get("height"),
                      "fps": got.get("fps"), "duration": got.get("duration") or info.get("duration")},
            "date": time.strftime("%Y-%m-%d %H:%M")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def collect(query: str, n: int, orientation: str | None = None, samples_dir: Path = SAMPLES_DIR) -> dict:
    """Cherche et télécharge jusqu'à n nouvelles vidéos ; renvoie les dossiers obtenus et les erreurs."""
    found = search(query, n, orientation)
    if not found:
        raise NoVideoFound(f"aucune nouvelle vidéo YouTube pour « {query} »"
                           + (f" ({orientation})" if orientation else "") + " : change de requête")
    dirs, errors = [], []
    for i, info in enumerate(found):
        time.sleep(random.uniform(*(DOWNLOAD_PAUSE if i else REQUEST_PAUSE)))  # après la recherche, puis entre deux
        try:
            dirs.append(download(info, query, samples_dir))
        except Exception as e:
            errors.append(f"{info['id']} : {str(e)[:120]}")
    return {"dirs": dirs, "errors": errors}


def load_meta(sample_dir: Path) -> dict:
    return json.loads((Path(sample_dir) / "meta.json").read_text(encoding="utf-8"))
