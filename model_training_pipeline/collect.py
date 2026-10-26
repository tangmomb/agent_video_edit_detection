"""Collecte directe d'un lot de vidéos YouTube variées, sans agents : téléchargement (une vidéo par
chaîne, chaînes absentes du dataset), puis un seul envoi en batch pour les trois annotateurs,
attente et récupération dans le même appel, comme generate_data.

    python -m model_training_pipeline collect --target 50 --max-usd 2.5 [--dry-run] [--no-wait]

--dry-run : cherche les vidéos (lecture des pages YouTube) sans rien télécharger ni annoter.
--no-wait : envoie les batchs et rend la main ; les récupérer plus tard avec
`python -m model_training_pipeline batch collect` (le PC peut être éteint entre-temps).
--direct : annotation en appels directs dès chaque téléchargement, sans batch (prix plein, deux fois
plus cher, mais résultats immédiats : un batch peut attendre des heures).
--orientation : seulement les requêtes de cette orientation.
--query "texte" (paysage) ou "texte|portrait", répétable : ces requêtes à la place de QUERIES.
Les vidéos déjà téléchargées mais sans tous leurs votes (collecte interrompue) sont reprises.
"""

import json
import random
import time

from video_source import youtube
from video_source.config import SAMPLES_DIR

from . import batch
from .frames import annotate, build

# Thèmes variés pour varier les styles de montage : vidéos très habillées (actualités, gaming,
# sport), sous-titrées (tutos, recettes, Shorts), sobres (vlogs, interviews, plans fixes).
# Portrait : Shorts du hashtag tiré de la requête (un ou deux mots courants).
QUERIES = [
    ("recette facile", "landscape"), ("vlog journée", "landscape"), ("tutoriel maquillage", "landscape"),
    ("résumé match football", "landscape"), ("bande annonce film", "landscape"),
    ("journal télévisé", "landscape"), ("unboxing smartphone", "landscape"), ("gameplay minecraft", "landscape"),
    ("clip musique live", "landscape"), ("interview rue", "landscape"), ("bricolage astuce", "landscape"),
    ("essai voiture", "landscape"), ("documentaire animaux", "landscape"), ("conférence science", "landscape"),
    ("cours guitare débutant", "landscape"), ("visite appartement", "landscape"), ("podcast extrait", "landscape"),
    ("météo bulletin", "landscape"), ("timelapse ville", "landscape"), ("drone paysage", "landscape"),
    ("recette", "portrait"), ("humour", "portrait"), ("voyage", "portrait"), ("football", "portrait"),
    ("maquillage", "portrait"), ("chat", "portrait"), ("fitness", "portrait"), ("danse", "portrait"),
    ("astuce", "portrait"), ("cuisine", "portrait"), ("moto", "portrait"), ("jardinage", "portrait"),
    ("skate", "portrait"), ("peinture", "portrait"), ("nature", "portrait"),
]


def _channels() -> set[str]:
    return {youtube.load_meta(p.parent).get("channel_id") for p in SAMPLES_DIR.glob("yt_*/meta.json")}


def collect(target: int = 50, per_query: int = 2, max_usd: float = 2.5, dry_run: bool = False,
            wait: bool = True, download_pause: tuple[float, float] = youtube.DOWNLOAD_PAUSE,
            direct: bool = False, only: str | None = None,
            queries: list[tuple[str, str]] | None = None) -> dict:
    avoid = _channels()
    got: list[tuple[str, str | None, object]] = []  # (requête, orientation, dossier ou info)
    asked: dict[tuple, int] = {}
    errors, est, cost = [], 0.0, 0.0
    stopped = None
    factor = 1 / batch.DISCOUNT if direct else 1.0  # estimations au tarif batch ; direct : prix plein

    def annotate_now(d) -> None:
        nonlocal cost
        res = annotate(d, max_usd=max_usd - cost)
        cost += res["cost_usd"]
        print(f"[collect] {d.name} annotée : {res['images_avec_accord']}/{res['images_annotees']} avec accord, "
              f"{res['cost_usd']:.3f} $ (total {cost:.2f} $)", flush=True)

    if not dry_run:  # reprise : vidéos téléchargées dont des votes manquent (hors batchs en cours)
        for p in sorted(SAMPLES_DIR.glob("yt_*/meta.json")):
            e = batch.estimate([p.parent])["total_usd_estime"] * factor
            if e and est + e <= max_usd:
                est += e
                got.append((youtube.load_meta(p.parent).get("query"), None, p.parent))
                if direct:
                    annotate_now(p.parent)
        if got:
            print(f"[collect] reprise de {len(got)} vidéo(s) déjà téléchargée(s) (≈ {est:.2f} $)", flush=True)
    for query, orientation in queries or QUERIES:
        if only and orientation != only:
            continue
        if len(got) >= target:
            break
        n = min(per_query, target - len(got))
        asked[(query, orientation)] = n
        try:
            found = youtube.search(query, n, orientation, avoid_channels=avoid)
        except Exception as e:
            errors.append(f"{query} : recherche en échec : {str(e)[:120]}")
            continue
        print(f"[collect] {query} ({orientation}) : {len(found)} vidéo(s)", flush=True)
        for info in found:
            if dry_run:
                got.append((query, orientation, info))
                continue
            time.sleep(random.uniform(*download_pause))
            try:
                d = youtube.download(info, query)
            except Exception as e:
                errors.append(f"{info['id']} : {str(e)[:120]}")
                continue
            e = batch.estimate([d])["total_usd_estime"] * factor  # extrait les images au passage
            if est + e > max_usd:
                stopped = f"plafond de {max_usd} $ atteint : {d.name} téléchargée mais pas annotée"
                break
            est += e
            got.append((query, orientation, d))
            print(f"[collect] {len(got)}/{target} {d.name} (≈ {e:.3f} $, total ≈ {est:.2f} $)", flush=True)
            if direct:
                annotate_now(d)
        if stopped:
            break
    if dry_run:
        return {"videos_trouvees": len(got), "chaines": len({i.get("channel_id") for _, _, i in got}),
                "par_requete": {q: sum(1 for x in got if x[0] == q) for q, _ in asked}, "erreurs": errors}

    dirs = [d for _, _, d in got]
    failed, jobs_all = 0, []
    # 2e passage pour les requêtes en échec, comme generate_data (direct : déjà annotées)
    for _ in range(2 if dirs and not direct else 0):
        sent = batch.submit(float("inf"), samples=dirs)["envoyes"]
        if not sent:
            break
        jobs = {s["job_id"] for s in sent}
        jobs_all += sorted(jobs)
        if not wait:
            stopped = stopped or ("batchs envoyés sans attendre : les récupérer avec "
                                  "python -m model_training_pipeline batch collect")
            break
        print(f"[collect] {len(jobs)} batch(s) envoyé(s), attente des résultats…", flush=True)
        batch.wait(jobs)
        res = batch.collect(job_ids=jobs)
        cost += res["cost_usd"]
        failed = sum(res["echecs"].values())
        if not failed:
            break
    build()

    # Historique des collectes (data_requested_history) : coût réparti au prorata des estimations
    per_query = {}
    for q, o, d in got:
        per_query.setdefault((q, o), []).append(d)
    incomplete = [d.name for d in dirs if batch.estimate([d])["total_usd_estime"]]  # votes manquants
    with open(SAMPLES_DIR.parent / "data_requests.jsonl", "a", encoding="utf-8") as fh:
        for (q, o), n in asked.items():
            ds = per_query.get((q, o), [])
            fh.write(json.dumps({"n": n, "ok": len(ds), "query": q, "orientation": o,
                                 "cost_usd": round(cost * len(ds) / max(1, len(dirs)), 4),
                                 "reason": "collecte directe (python -m model_training_pipeline collect)",
                                 "time": time.time()}) + "\n")
    return {"videos_collectees": len(dirs), "cout_annotation_usd": round(cost, 4),
            "cout_estime_usd": round(est, 4), "requetes_en_echec": failed, "batchs": jobs_all,
            "videos_avec_votes_manquants": incomplete, "arret": stopped,
            "erreurs": errors}
