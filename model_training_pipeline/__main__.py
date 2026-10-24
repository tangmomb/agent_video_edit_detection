import argparse

from .frames import FPS, build


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m model_training_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("frames", help="extrait les frames des vidéos et réécrit l'index (sans appel payant)")
    b.add_argument("--fps", type=float, default=FPS)
    a = sub.add_parser("annotate", help="fait voter les 3 LLM sur les images pas encore annotées (payant)")
    a.add_argument("--max-usd", type=float, required=True, help="plafond de dépense pour l'ensemble des vidéos")
    bt = sub.add_parser("batch", help="annotation par les API batch (50 %% moins cher, résultats différés)")
    bt.add_argument("action", choices=["estimate", "submit", "status", "collect"])
    bt.add_argument("--max-usd", type=float, help="submit : plafond du coût estimé")
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
    elif args.cmd == "batch":
        import json

        from . import batch
        if args.action == "submit" and args.max_usd is None:
            p.error("batch submit demande --max-usd")
        out = {"estimate": batch.estimate, "status": batch.status, "collect": batch.collect,
               "submit": lambda: batch.submit(args.max_usd)}[args.action]()
        print(json.dumps(out, indent=2, ensure_ascii=False))
    elif args.cmd == "info":
        import json

        from .lab import Lab
        print(json.dumps(Lab().dataset_info(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
