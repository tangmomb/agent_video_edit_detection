import argparse

from .frames import FPS, build


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m model_training_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("frames", help="extrait et étiquette les frames des échantillons")
    b.add_argument("--fps", type=float, default=FPS)
    sub.add_parser("info", help="état du dataset (outil dataset_info de l'agent)")
    args = p.parse_args()

    if args.cmd == "frames":
        print(build(fps=args.fps))
    elif args.cmd == "info":
        import json

        from .lab import Lab
        print(json.dumps(Lab().dataset_info(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
