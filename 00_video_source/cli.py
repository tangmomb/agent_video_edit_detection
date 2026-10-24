import argparse
import sys
from pathlib import Path

from .config import SAMPLES_DIR
from .preview import contact_sheet
from .youtube import collect


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="video_source")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="Télécharge N nouvelles vidéos YouTube (sans les annoter)")
    c.add_argument("--query", required=True, help="recherche YouTube")
    c.add_argument("--n", type=int, default=1)
    c.add_argument("--orientation", choices=["portrait", "landscape"])
    c.add_argument("--out", type=Path, default=SAMPLES_DIR)

    pv = sub.add_parser("preview", help="Planche de contrôle avec les étiquettes des annotateurs")
    pv.add_argument("sample_dir", type=Path)

    args = p.parse_args(argv)

    if args.cmd == "collect":
        got = collect(args.query, args.n, args.orientation, args.out)
        for d in got["dirs"]:
            print(d)
        for e in got["errors"]:
            print(f"échec : {e}", file=sys.stderr)
        return 0

    if args.cmd == "preview":
        print(contact_sheet(args.sample_dir))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
