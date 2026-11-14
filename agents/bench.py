"""Banc d'agents : compare des modèles LLM (le même pour les quatre agents) sur un scénario figé, sans
jamais toucher aux vraies données.

    python -m agents.bench snapshot --name b1 [--hidden 0.25]   # instantané figé du dataset (gratuit)
    python -m agents.bench run --name b1 --runs 3 --budget 1.5  # runs d'agents (PAYANT)
    python -m agents.bench score --name b1                      # notation et tableau (gratuit)

Tout vit dans bench/<nom>/ :
- full/data : copie des vidéos entièrement annotées du dataset (votes, images, index, embeddings) ;
- hidden.json : chaînes cachées (une part par orientation), que les agents ne voient jamais ;
- base/data : full sans les chaînes cachées, modèle de départ de chaque run ;
- runs/<modèle>_<k>/ : dossier de travail d'un run (AGENT_LAB_DIR : data, model_results, runs).

Chaque run part de base/data, sans collecte (--data-budget 0), avec le même budget pour tous : un
budget large teste la qualité pure, le coût réel est noté à côté. La note vient du jeu caché : le
modèle de production choisi en fin de run (production.json) y est mesuré au seuil fixé sur val,
avec les mêmes embeddings qu'à l'entraînement : rappel et précision à ce seuil (la précision
visée tient-elle sur des chaînes jamais vues ?), AP, et rappel idéal à 90 % (meilleur seuil
possible sur le jeu caché : qualité du modèle sans l'erreur de seuil). Le comportement vient de pipeline.json (coût,
durée, outils en échec, expériences, test final). Les images sont des liens physiques vers le
dataset (jamais réécrites) ; les fichiers texte, réécrits par le labo, sont copiés.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from video_source.config import ROOT

BENCH_DIR = ROOT / "bench"
MODELS = ["anthropic/claude-haiku-5-5", "mistral/deepseek-v4-1-flash-1017", "mistral/mistral-large-4-0", "mistral/mistral-large-2512",
          "mistral/zai-glm-5-3-flash", "openai/gpt-5.6-luna", "openai/gpt-6-luna"]
GOAL = "Lance une session d'amélioration du classifieur d'ajouts de montage."
TARGET = 0.9  # précision visée (metrics.DEFAULT_PRECISION, sans importer sklearn ici)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_data(src: Path, dst: Path, samples: set[str]) -> None:
    """Copie le dataset de src pour ces vidéos : images (.jpg) et embeddings (.npz) en liens
    physiques, car le labo ne réécrit jamais les images et remplace un cache d'embeddings complété
    par un fichier neuf ; le reste copié (votes.json compris, réécrit à chaque annotation)."""
    for sid in samples:
        for f in ("meta.json",):  # votes.json : avec les images, copié ci-dessous
            p = src / "videos_samples" / sid / f
            if p.exists():
                (dst / "videos_samples" / sid).mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst / "videos_samples" / sid / f)
        for p in (src / "videos_samples_frames" / "img" / sid).iterdir():
            out = dst / "videos_samples_frames" / "img" / sid / p.name
            if p.suffix == ".jpg":
                _link_or_copy(p, out)
            else:  # .extraction
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
    feat = src / "videos_samples_frames" / "feat"
    for cdir in feat.iterdir() if feat.exists() else []:
        for p in cdir.iterdir():
            out = dst / "videos_samples_frames" / "feat" / cdir.name / p.name
            if p.suffix == ".npz" and p.stem in samples:
                _link_or_copy(p, out)
            elif p.suffix != ".npz":
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
    lines = (src / "videos_samples_frames" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    keep = [line for line in lines if json.loads(line)["sample_id"] in samples]
    (dst / "videos_samples_frames" / "index.jsonl").write_text("".join(l + "\n" for l in keep), encoding="utf-8")


def snapshot(name: str, hidden: float = 0.25, seed: int = 0, src: Path = ROOT / "data") -> dict:
    """Instantané figé : vidéos entièrement annotées, et une part des chaînes cachée aux agents."""
    from model_training_pipeline import frames
    out = BENCH_DIR / name
    if out.exists():
        raise SystemExit(f"{out} existe déjà : choisis un autre nom")
    rows = [json.loads(l) for l in (src / "videos_samples_frames" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = {r["sample_id"] for r in rows}
    # Vidéos à qui il manque des votes (encore en annotation…) : écartées, étiquettes incomplètes.
    # Même règle que l'annotation (frames.pending) : les images sautées comme quasi identiques ne sont
    # jamais votées, elles ne rendent pas une vidéo incomplète.
    incomplete = {s for s in samples if frames.pending(src / "videos_samples" / s, src / "videos_samples_frames")}
    samples -= incomplete
    by_channel: dict[str, set[str]] = defaultdict(set)
    orient: dict[str, str] = {}
    for r in rows:
        if r["sample_id"] in samples:
            by_channel[r["source"]].add(r["sample_id"])
    for ch, sids in by_channel.items():
        meta = json.loads((src / "videos_samples" / sorted(sids)[0] / "meta.json").read_text(encoding="utf-8"))
        orient[ch] = "portrait" if meta["video"]["height"] > meta["video"]["width"] else "paysage"
    rng = random.Random(seed)
    hidden_ch = []
    for o in ("paysage", "portrait"):  # une part des chaînes de chaque orientation
        chs = sorted(c for c in by_channel if orient[c] == o)
        rng.shuffle(chs)
        hidden_ch += chs[:round(hidden * len(chs))]
    hidden_s = {s for c in hidden_ch for s in by_channel[c]}
    _copy_data(src, out / "full" / "data", samples)
    _copy_data(src, out / "base" / "data", samples - hidden_s)
    info = {"seed": seed, "hidden_ratio": hidden, "chaines_cachees": sorted(hidden_ch),
            "videos_cachees": sorted(hidden_s), "videos_visibles": len(samples - hidden_s),
            "chaines_visibles": len(by_channel) - len(hidden_ch),
            "videos_ecartees_annotation_incomplete": sorted(incomplete)}
    (out / "hidden.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: (len(v) if isinstance(v, list) else v) for k, v in info.items()}


def _slug(model: str) -> str:
    return model.replace("/", "__")


def run(name: str, models: list[str], runs: int, budget: float, final_tests: int = 1, goal: str = GOAL) -> None:
    """Runs d'agents, l'un après l'autre ; un run déjà terminé (pipeline.json « done ») est sauté."""
    from agents.runner import price
    for m in models:
        price(m)  # tarif inconnu : erreur avant tout appel payant
    base = BENCH_DIR / name / "base" / "data"
    if not base.exists():
        raise SystemExit(f"pas d'instantané {name} : python -m agents.bench snapshot --name {name}")
    samples = {p.name for p in (base / "videos_samples").iterdir()}
    for k in range(1, runs + 1):  # un tour de tous les modèles, puis le suivant : écarts dans le temps répartis
        for m in models:
            wdir = BENCH_DIR / name / "runs" / f"{_slug(m)}_{k}"
            done = [p for p in wdir.glob("runs/*/pipeline.json")
                    if json.loads(p.read_text(encoding="utf-8"))["run"]["status"] == "done"]
            if done:
                print(f"[bench] {m} #{k} : déjà fait", flush=True)
                continue
            shutil.rmtree(wdir, ignore_errors=True)  # run interrompu : on repart de zéro
            _copy_data(base, wdir / "data", samples)
            print(f"[bench] {m} #{k} : en cours ({wdir})", flush=True)
            env = {**os.environ, "AGENT_LAB_DIR": str(wdir), "PYTHONIOENCODING": "utf-8"}
            with open(wdir / "console.log", "w", encoding="utf-8") as log:
                code = subprocess.call([sys.executable, "-m", "agents", "--goal", goal, "--budget-usd", str(budget),
                                        "--data-budget", "0", "--annotation-usd", "0",
                                        "--final-tests", str(final_tests), "--agent-model", m],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f"[bench] {m} #{k} : terminé (code {code})", flush=True)


def _score_one(exp_dir: str, threshold: float, hidden: list[str]) -> dict:
    """Mesure d'un modèle sur le jeu caché (lancé avec AGENT_LAB_DIR = full : mêmes embeddings)."""
    import pickle

    import numpy as np

    from model_training_pipeline import metrics
    from model_training_pipeline.lab import Lab
    d = Path(exp_dir)
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    with open(d / "model.pkl", "rb") as fh:
        clf = pickle.load(fh)
    lab = Lab()
    hid = set(hidden)
    idx = np.array([i for i, r in enumerate(lab.rows) if r["sample_id"] in hid and r["label_any"] is not None])
    X, _ = lab._X(config["features"], idx)
    y = lab._y()[idx]
    probs = clf.predict_proba(X[idx])[:, 1]
    s = metrics.summary(y, probs)
    ideal = metrics.recall_at_precision(y, probs, metrics.DEFAULT_PRECISION)
    return {"au_seuil_val": metrics.operating_point(y, probs, threshold), "auc": s["auc"], "ap": s["ap"],
            "rappel_ideal": ideal["recall"] if ideal else 0.0, "images": len(y), "positifs": int(y.sum())}


def _trace(wdir: Path) -> dict:
    """Comportement d'un run, d'après pipeline.json."""
    p = next(iter(sorted(wdir.glob("runs/*/pipeline.json"))), None)
    if p is None:
        return {"statut": "absent"}
    t = json.loads(p.read_text(encoding="utf-8"))
    run, steps = t["run"], t["steps"]
    tools = [s for s in steps if s["type"] == "tool"]
    failed = [s for s in tools if s["error"] or (isinstance(s["result"], dict) and s["result"].get("error"))]
    from datetime import datetime
    dur = ((datetime.fromisoformat(run["ended"]) - datetime.fromisoformat(run["started"])).total_seconds() / 60
           if run["ended"] else None)
    return {"statut": run["status"], "cout_usd": run["cost_usd"], "duree_min": round(dur, 1) if dur else None,
            "tokens": run["tokens"], "appels_agents": run["agent_calls"], "outils": len(tools),
            "outils_en_echec": len(failed), "experiences": sum(1 for s in tools if s["tool"] == "train_model"),
            "tests_finaux": sum(1 for s in tools if s["tool"] == "final_test"), "erreur": run.get("error")}


def score(name: str) -> list[dict]:
    root = BENCH_DIR / name
    hidden = json.loads((root / "hidden.json").read_text(encoding="utf-8"))["videos_cachees"]
    env = {**os.environ, "AGENT_LAB_DIR": str(root / "full"), "PYTHONIOENCODING": "utf-8"}
    out = []
    for wdir in sorted((root / "runs").iterdir()) if (root / "runs").exists() else []:
        model, _, k = wdir.name.rpartition("_")
        r = {"modele": model.replace("__", "/"), "run": int(k), **_trace(wdir)}
        prod_path = wdir / "model_results" / "production.json"
        if prod_path.exists():
            prod = json.loads(prod_path.read_text(encoding="utf-8"))
            r["production"] = {"experiment_id": prod["experiment_id"], "seuil": prod["threshold"],
                               "val_recall": prod["val_recall"], "val_precision": prod["val_precision"],
                               "test": prod["test"]}
            res = subprocess.run([sys.executable, "-m", "agents.bench", "_score",
                                  str(wdir / "model_results" / prod["experiment_id"]), str(prod["threshold"])],
                                 cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
            r["cache"] = json.loads(res.stdout.strip().splitlines()[-1]) if res.returncode == 0 else \
                {"erreur": res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "?"}
        out.append(r)
    (root / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "results.md").write_text(_table(out), encoding="utf-8")
    return out


def _table(results: list[dict]) -> str:
    def f(v, nd=3):
        return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
    head = ("| Modèle | Run | Statut | AP caché | Rappel caché | Précision cachée | Rappel idéal | Rappel val | "
            "Test final | Expériences | Outils en échec | Coût $ | Durée min |\n|" + "---|" * 13 + "\n")
    lines = []
    for r in sorted(results, key=lambda r: (r["modele"], r["run"])):
        c, p = r.get("cache") or {}, r.get("production") or {}
        op = c.get("au_seuil_val") or {}
        lines.append(f"| {r['modele']} | {r['run']} | {r['statut']} | {f(c.get('ap'))} | {f(op.get('recall'))} | "
                     f"{f(op.get('precision'))} | {f(c.get('rappel_ideal'))} | {f(p.get('val_recall'))} | {(p.get('test') or {}).get('statut', '—')} | "
                     f"{f(r.get('experiences'))} | {f(r.get('outils_en_echec'))} | {f(r.get('cout_usd'), 2)} | "
                     f"{f(r.get('duree_min'), 1)} |")
    return _summary(results) + "\n" + head + "\n".join(lines) + "\n"


def _summary(results: list[dict]) -> str:
    """Moyenne ± écart-type par modèle sur ses runs notés (un seul run ne départage rien)."""
    import statistics as st

    def ms(vals: list[float], nd: int = 3) -> str:
        if not vals:
            return "—"
        sd = f" ± {st.stdev(vals):.{nd}f}" if len(vals) > 1 else ""
        return f"{st.mean(vals):.{nd}f}{sd}"
    head = ("| Modèle | Runs notés | AP caché | Rappel caché | Précision cachée | Précision tenue | "
            "Rappel idéal | Tests finaux réussis | Coût $ | Durée min |\n|" + "---|" * 10 + "\n")
    by: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by[r["modele"]].append(r)
    rows = []
    for m, rs in by.items():
        scored = [r for r in rs if (r.get("cache") or {}).get("ap") is not None]
        ops = [r["cache"]["au_seuil_val"] for r in scored]
        rows.append((st.mean(r["cache"]["ap"] for r in scored) if scored else -1, m,
                     f"| {m} | {len(scored)}/{len(rs)} | {ms([r['cache']['ap'] for r in scored])} | "
                     f"{ms([o['recall'] for o in ops])} | {ms([o['precision'] for o in ops])} | "
                     f"{sum(o['precision'] >= TARGET for o in ops)}/{len(ops)} | "
                     f"{ms([r['cache']['rappel_ideal'] for r in scored if 'rappel_ideal' in r['cache']])} | "
                     f"{sum((r.get('production') or {}).get('test', {}).get('statut') == 'réussi' for r in rs)}/{len(rs)} | "
                     f"{ms([r['cout_usd'] for r in rs if r.get('cout_usd') is not None], 2)} | "
                     f"{ms([r['duree_min'] for r in rs if r.get('duree_min')], 1)} |"))
    return ("## Par modèle (moyenne ± écart-type)\n\nPrécision tenue : runs dont la précision cachée atteint "
            f"{TARGET}. Rappel idéal : rappel à {TARGET} de précision au meilleur seuil possible sur le jeu "
            "caché (qualité du modèle, sans l'erreur de seuil).\n\n" + head
            + "\n".join(r for _, _, r in sorted(rows, reverse=True)) + "\n\n## Par run\n\n")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m agents.bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="instantané figé du dataset, avec un jeu caché (gratuit)")
    s.add_argument("--name", required=True)
    s.add_argument("--hidden", type=float, default=0.25, help="part des chaînes cachées aux agents")
    s.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run", help="runs d'agents (PAYANT)")
    r.add_argument("--name", required=True)
    r.add_argument("--runs", type=int, default=5, help="runs par modèle (3 ne départagent que les gros écarts)")
    r.add_argument("--budget", type=float, required=True, help="budget agents de chaque run, en $ (le même pour tous)")
    r.add_argument("--models", nargs="+", default=MODELS)
    r.add_argument("--final-tests", type=int, default=1)
    sc = sub.add_parser("score", help="note les runs sur le jeu caché et écrit le tableau (gratuit)")
    sc.add_argument("--name", required=True)
    one = sub.add_parser("_score")  # interne : lancé par score avec AGENT_LAB_DIR = full
    one.add_argument("exp_dir")
    one.add_argument("threshold", type=float)
    a = p.parse_args()
    if a.cmd == "snapshot":
        print(json.dumps(snapshot(a.name, a.hidden, a.seed), indent=2, ensure_ascii=False))
    elif a.cmd == "run":
        run(a.name, a.models, a.runs, a.budget, a.final_tests)
    elif a.cmd == "score":
        print(_table(score(a.name)))
    else:
        from video_source.config import LAB_DIR
        root = LAB_DIR.parent  # bench/<nom>/full -> bench/<nom>
        hidden = json.loads((root / "hidden.json").read_text(encoding="utf-8"))["videos_cachees"]
        print(json.dumps(_score_one(a.exp_dir, a.threshold, hidden)))


if __name__ == "__main__":
    main()
