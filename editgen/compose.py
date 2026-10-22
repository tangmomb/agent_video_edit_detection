"""Composition ffmpeg : vidéo source + overlays PNG animés -> mp4."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg

from .config import ffmpeg_bin


@dataclass
class Overlay:
    png: Path
    x: int  # coin haut-gauche du PNG, position au repos
    y: int
    t_start: float
    t_end: float
    anim_in: dict
    anim_out: dict


def probe(path: Path) -> dict:
    """Renvoie {'width', 'height', 'duration', 'fps'} (rotation appliquée)."""
    gen = imageio_ffmpeg.read_frames(str(path))
    try:
        meta = next(gen)
    finally:
        gen.close()
    w, h = meta["size"]
    if int(meta.get("rotate") or 0) % 180 == 90:
        w, h = h, w
    return {"width": w, "height": h, "duration": float(meta["duration"]), "fps": float(meta["fps"])}


def _pos_expr(rest: int, offset: float, ov: Overlay) -> str:
    ain = ov.anim_in
    if not ain["type"].startswith("slide") or not offset:
        return str(rest)
    p = f"clip((t-{ov.t_start})/{ain['duration']},0,1)"
    return f"'{rest}+({offset:.2f})*pow(1-{p},3)'"


def compose(video_in: Path, out_path: Path, overlays: list[Overlay], width: int, height: int,
            fps: int, ss: float, duration: float, crf: int = 22) -> None:
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{ss:.3f}", "-t", f"{duration:.3f}", "-i", str(video_in)]
    for ov in overlays:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{duration:.3f}", "-i", str(ov.png)]

    chains = [f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
              f"crop={width}:{height},setsar=1,fps={fps},format=yuv420p[b0]"]
    for i, ov in enumerate(overlays, start=1):
        fx = ["format=rgba"]
        ain, aout = ov.anim_in, ov.anim_out
        if ain["type"] == "fade" or ain.get("fade"):
            fx.append(f"fade=t=in:st={ov.t_start}:d={ain['duration']}:alpha=1")
        if aout["type"] == "fade":
            fx.append(f"fade=t=out:st={ov.t_end - aout['duration']:.3f}:d={aout['duration']}:alpha=1")
        chains.append(f"[{i}:v]{','.join(fx)}[o{i}]")
        dx, dy = ain.get("offset", (0, 0))
        chains.append(
            f"[b{i - 1}][o{i}]overlay=x={_pos_expr(ov.x, dx, ov)}:y={_pos_expr(ov.y, dy, ov)}"
            f":eval=frame:enable='gte(t,{ov.t_start})*lt(t,{ov.t_end})'[b{i}]"
        )

    cmd += ["-filter_complex", ";".join(chains), "-map", f"[b{len(overlays)}]", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg a échoué :\n{res.stderr[-2000:]}")
