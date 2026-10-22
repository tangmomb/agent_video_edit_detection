"""Format de vérité terrain et helpers pour l'interroger à un instant t.

Un élément annoté :
{
  "id": "el_03", "type": "keyword|question|subtitle|lower_third|image|shape",
  "content": "ASTUCE" | {...},
  "t_start": 1.2, "t_end": 3.8,                      # secondes, dans la vidéo produite
  "bbox": [x, y, w, h],                              # position au repos, pixels
  "animation": {
    "in":  {"type": "none|fade|slide_left|slide_right|slide_up|slide_down",
            "duration": 0.3, "offset": [dx, dy], "fade": true},
    "out": {"type": "none|fade", "duration": 0.3}
  },
  "style": {...}, "group": "sub_1" (optionnel)
}
Pendant un slide-in, la bbox est décalée de offset * (1 - p)^3 (ease-out cubique).
"""

import json
from pathlib import Path

SCHEMA_VERSION = 1


def _progress(t: float, start: float, duration: float) -> float:
    if duration <= 0:
        return 1.0
    return min(max((t - start) / duration, 0.0), 1.0)


def is_visible(el: dict, t: float) -> bool:
    return el["t_start"] <= t < el["t_end"]


def bbox_at(el: dict, t: float) -> list[float] | None:
    if not is_visible(el, t):
        return None
    x, y, w, h = el["bbox"]
    ain = el["animation"]["in"]
    if ain["type"].startswith("slide"):
        dx, dy = ain["offset"]
        k = (1 - _progress(t, el["t_start"], ain["duration"])) ** 3
        return [x + dx * k, y + dy * k, w, h]
    return [x, y, w, h]


def opacity_at(el: dict, t: float) -> float:
    if not is_visible(el, t):
        return 0.0
    op = 1.0
    ain, aout = el["animation"]["in"], el["animation"]["out"]
    if ain["type"] == "fade" or ain.get("fade"):
        op = min(op, _progress(t, el["t_start"], ain["duration"]))
    if aout["type"] == "fade":
        op = min(op, _progress(el["t_end"] - t, 0.0, aout["duration"]))
    return op


def visible_elements(ann: dict, t: float, min_opacity: float = 0.05) -> list[dict]:
    return [el for el in ann["elements"] if opacity_at(el, t) >= min_opacity]


def load(sample_dir: str | Path) -> dict:
    return json.loads((Path(sample_dir) / "annotations.json").read_text(encoding="utf-8"))


def save(sample_dir: str | Path, ann: dict) -> Path:
    path = Path(sample_dir) / "annotations.json"
    path.write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
