"""Découverte des polices. Sur un serveur, déposer des .ttf dans assets/fonts/."""

from functools import lru_cache
from pathlib import Path

from .config import ASSETS_DIR

SYSTEM_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
]

# Polices système « sûres » (latin + accents) qu'on accepte si présentes.
PREFERRED = [
    "arialbd", "ariblk", "impact", "segoeuib", "seguibl", "verdanab", "tahomabd",
    "calibrib", "trebucbd", "georgiab", "comicbd", "bahnschrift",
    "dejavusans-bold", "liberationsans-bold", "roboto-bold", "roboto-black",
    "montserrat", "anton", "bebasneue", "poppins", "oswald", "opensans-bold",
]


@lru_cache(maxsize=1)
def available_fonts() -> tuple[Path, ...]:
    found: list[Path] = []
    custom = ASSETS_DIR / "fonts"
    if custom.exists():
        found += sorted(p for p in custom.rglob("*") if p.suffix.lower() in (".ttf", ".otf"))
    for d in SYSTEM_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in (".ttf", ".otf") and any(p.stem.lower().startswith(s) for s in PREFERRED):
                found.append(p)
    # Les graisses fines deviennent illisibles avec un contour : on les écarte.
    found = [p for p in found if not any(w in p.stem.lower() for w in ("thin", "light"))]
    if not found:
        raise RuntimeError("Aucune police trouvée : ajoutez des .ttf dans assets/fonts/.")
    return tuple(dict.fromkeys(found))
