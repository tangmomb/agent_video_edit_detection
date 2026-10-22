import argparse
import json
import sys
from pathlib import Path

from .config import SAMPLES_DIR
from .generator import GenerationConfig, generate_sample
from .preview import contact_sheet
from .sources import default_photo_source, default_video_source


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="editgen")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Génère N échantillons annotés")
    g.add_argument("--n", type=int, default=1)
    g.add_argument("--seed", type=int, default=0, help="seed du premier échantillon (puis +1)")
    g.add_argument("--query", help="requête de recherche en anglais (sinon tirée au hasard)")
    g.add_argument("--orientation", choices=["portrait", "landscape"])
    g.add_argument("--local-video", type=Path, help="utiliser une vidéo locale au lieu d'une banque de vidéos")
    g.add_argument("--lang", choices=["fr", "en", "mix"], default="fr")
    g.add_argument("--max-duration", type=float, default=12.0)
    g.add_argument("--out", type=Path, default=SAMPLES_DIR)
    g.add_argument("--preview", action="store_true", help="écrit aussi preview.jpg")

    pv = sub.add_parser("preview", help="Planche de contrôle avec les bbox")
    pv.add_argument("sample_dir", type=Path)

    args = p.parse_args(argv)

    if args.cmd == "generate":
        cfg = GenerationConfig(lang=args.lang, max_duration=args.max_duration)
        videos = photos = None
        for i in range(args.n):
            seed = args.seed + i
            try:
                if videos is None and not args.local_video:
                    videos = default_video_source()
                if photos is None:
                    photos = default_photo_source()
                out = generate_sample(seed, args.out, cfg, query=args.query, orientation=args.orientation,
                                      video_source=videos, photo_source=photos, local_video=args.local_video)
            except Exception as e:
                print(f"[seed {seed}] échec : {e}", file=sys.stderr)
                continue
            n_el = len(json.loads((out / "annotations.json").read_text(encoding="utf-8"))["elements"])
            print(f"[seed {seed}] {out}  ({n_el} éléments)")
            if args.preview:
                contact_sheet(out)
        return 0

    if args.cmd == "preview":
        print(contact_sheet(args.sample_dir))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
