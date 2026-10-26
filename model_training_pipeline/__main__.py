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
    c = sub.add_parser("collect", help="collecte directe de vidéos variées puis annotation en batch (payant)")
    c.add_argument("--target", type=int, default=50, help="vidéos à collecter")
    c.add_argument("--per-query", type=int, default=2, help="vidéos au plus par requête")
    c.add_argument("--max-usd", type=float, required=True, help="plafond du coût d'annotation estimé")
    c.add_argument("--dry-run", action="store_true", help="cherche seulement : ni téléchargement ni annotation")
    c.add_argument("--no-wait", action="store_true",
                   help="envoie les batchs sans attendre (récupération plus tard : batch collect)")
    c.add_argument("--pause", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="attente aléatoire entre deux téléchargements, en s (défaut : 60 180)")
    c.add_argument("--direct", action="store_true",
                   help="annotation en appels directs dès chaque téléchargement (prix plein, sans attente)")
    c.add_argument("--orientation", choices=["portrait", "landscape"], help="seulement ces requêtes")
    c.add_argument("--query", action="append", metavar="TEXTE[|portrait]",
                   help="requête à la place de la liste par défaut (répétable ; paysage sauf |portrait)")
    args = p.parse_args()

    if args.cmd == "collect":
        import json

        from video_source.youtube import DOWNLOAD_PAUSE

        from .collect import collect
        print(json.dumps(collect(args.target, args.per_query, args.max_usd, args.dry_run, not args.no_wait,
                                 tuple(args.pause or DOWNLOAD_PAUSE), args.direct, args.orientation,
                                 [(q.partition("|")[0], q.partition("|")[2] or "landscape") for q in args.query]
                                 if args.query else None),
                         indent=2, ensure_ascii=False))
        return

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
