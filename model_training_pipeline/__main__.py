import argparse

from .frames import FPS, build


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m model_training_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("frames", help="extrait les frames des vidéos et réécrit l'index (sans appel payant)")
    b.add_argument("--fps", type=float, default=FPS)
    a = sub.add_parser("annotate", help="fait voter les 3 LLM sur les images pas encore annotées (payant)")
    a.add_argument("--max-usd", type=float, required=True, help="plafond de dépense pour l'ensemble des vidéos")
    sub.add_parser("info", help="état du dataset (outil dataset_info de l'agent)")
    args = p.parse_args()

    if args.cmd == "frames":
        print(build(fps=args.fps))
    elif args.cmd == "annotate":
        from video_source.config import SAMPLES_DIR

        from .frames import annotate
        left = args.max_usd
        for sdir in sorted(SAMPLES_DIR.glob("yt_*")):
            if left <= 0:
                print("plafond atteint")
                break
            res = annotate(sdir, max_usd=left)
            left -= res["cost_usd"]
            print(res)
        print(build())
    elif args.cmd == "info":
        import json

        from .lab import Lab
        print(json.dumps(Lab().dataset_info(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
