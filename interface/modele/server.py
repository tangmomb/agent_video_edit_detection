"""Interface de test du modèle : on envoie une vidéo, on voit les frames où il détecte un ajout.

    .venv/Scripts/python.exe interface/server.py [--exp exp_037] [--port 8780]

Sans --exp, utilise le modèle désigné dans model_results/production.json (et son seuil), rechargé
quand ce fichier change. La vidéo
est traitée exactement comme à l'entraînement : frames à FPS par seconde (frames.extract), mêmes
embeddings (features, lab._X), même classifieur.
Fichiers de travail dans interface/modele/_jobs/<id>/ (frames + résultat), effaçables à tout moment.
Aucun appel aux LLM : l'annotation a été retirée de cette page (03/11/2026). Avec EMBED_GPU=modal,
les embeddings se calculent sur le GPU distant (quelques centimes par vidéo), et sur place s'il ne
répond pas à temps.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model_training_pipeline import metrics  # noqa: E402
from model_training_pipeline.features import _encoder, embed_images, embed_modal  # noqa: E402
from model_training_pipeline.frames import FPS, extract  # noqa: E402
from model_training_pipeline.lab import EXP_DIR  # noqa: E402
from interface import dev_reload  # noqa: E402
from interface.quota import Quota, Rejected  # noqa: E402

HERE = Path(__file__).resolve().parent
JOBS = HERE / "_jobs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_WAIT_S = 120  # attente maximale du GPU distant par morceau de vidéo, avant le calcul sur place
_gpu = Lock()  # une analyse à la fois sur le GPU
# Page publique : au plus 5 vidéos par jour sur le GPU distant, quelques centimes chacune.
QUOTA = Quota(JOBS, per_ip=None, per_day=5, max_mb=500, max_s=300)


def _remote() -> bool:
    return os.environ.get("EMBED_GPU") == "modal"


def _embed(frames: list[Path], extractor: str, view: str) -> tuple[np.ndarray, str]:
    """Les embeddings, et où ils ont été calculés."""
    if _remote():
        try:
            return embed_modal(frames, extractor, view, wait_s=GPU_WAIT_S), "modal"
        except Exception as e:  # quelle que soit la panne, la page doit répondre
            print(f"GPU distant indisponible, calcul sur place : {type(e).__name__}: {e}", flush=True)
    with torch.inference_mode():
        return embed_images(frames, extractor, view, batch_size=32), DEVICE


class Model:
    """Classifieur d'une expérience + son seuil, prêt à noter les frames d'une vidéo."""

    def __init__(self, exp_id: str | None):
        prod = json.loads((EXP_DIR / "production.json").read_text(encoding="utf-8")) \
            if (EXP_DIR / "production.json").exists() else {}
        self.exp_id = exp_id or prod.get("experiment_id")
        if not self.exp_id:
            raise SystemExit("aucun modèle en production : passe --exp exp_XXX")
        d = EXP_DIR / self.exp_id
        self.config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        self.metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        self.final_test = json.loads((d / "final_test.json").read_text(encoding="utf-8")) \
            if (d / "final_test.json").exists() else None
        with open(d / "model.pkl", "rb") as fh:
            self.clf = pickle.load(fh)
        if prod.get("experiment_id") == self.exp_id:
            self.threshold, self.precision = prod["threshold"], prod["precision_visee"]
        else:  # seuil de l'expérience à la précision par défaut, choisi sur val comme dans le labo
            self.precision = metrics.DEFAULT_PRECISION
            op = self.metrics["val"].get(f"rappel_a_precision_{self.precision}")
            if not isinstance(op, dict):  # pas atteinte à l'entraînement : recalcul (seuil prudent si chaînes connues)
                preds = json.loads((d / "val_preds.json").read_text(encoding="utf-8"))
                op = metrics.recall_at_precision(np.array(preds["y"]), np.array(preds["probs"]), self.precision,
                                                 preds.get("groups"))
            self.threshold = op["threshold"]
        if not _remote():  # charge les encodeurs au démarrage, pas à la 1re vidéo
            for spec in self.config["features"]:
                _encoder(spec.partition(":")[0], DEVICE)

    def info(self) -> dict:
        ft = (self.final_test or {}).get("test")
        return {"experiment_id": self.exp_id, "features": self.config["features"],
                "model": self.config["model"],
                "split_id": self.config.get("split_id"), "threshold": self.threshold,
                "precision_visee": self.precision, "device": "modal" if _remote() else DEVICE,
                "test": {k: ft[k] for k in ("precision", "recall")} if ft else None}

    def score(self, frames: list[Path]) -> tuple[np.ndarray, str]:
        """Les probabilités, et où les embeddings ont été calculés (sur place si un seul l'a été)."""
        parts = [_embed(frames, ext, view or "full")
                 for ext, _, view in (s.partition(":") for s in self.config["features"])]
        device = next((d for _, d in parts if d != "modal"), "modal")
        return self.clf.predict_proba(np.concatenate([x for x, _ in parts], axis=1))[:, 1], device


def analyze(model: Model, video: Path, job: Path) -> dict:
    t0 = time.perf_counter()
    frames = extract(video, job / "frames", FPS)  # mêmes réglages que frames.build
    if not frames:
        raise ValueError("aucune frame extraite : fichier vidéo illisible ?")
    with _gpu:
        probs, device = model.score(frames)
    result = {"job": job.name, "fps": FPS, "model": {**model.info(), "device": device},
              "duration_s": round(time.perf_counter() - t0, 2),
              "frames": [{"t": round(i / FPS, 3), "prob": round(float(p), 5),
                          "url": f"jobs/{job.name}/frames/{f.name}"}
                         for i, (f, p) in enumerate(zip(frames, probs))]}
    (job / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def _prod_mtime() -> float | None:
    path = EXP_DIR / "production.json"
    return path.stat().st_mtime if path.exists() else None


class Handler(SimpleHTTPRequestHandler):
    model: Model
    exp_forced: str | None = None  # --exp : modèle fixe, production.json ignoré
    prod_mtime: float | None = None
    _reload = Lock()

    @classmethod
    def current_model(cls) -> Model:
        """Le modèle à jour : rechargé quand production.json change (fin de run), sans redémarrer."""
        if cls.exp_forced is None and _prod_mtime() != cls.prod_mtime:
            with cls._reload:
                mtime = _prod_mtime()
                if mtime != cls.prod_mtime:
                    cls.model = Model(None)
                    cls.prod_mtime = mtime
                    print(f"production.json modifié : modèle {cls.model.exp_id} chargé", flush=True)
        return cls.model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("[interface] " + fmt % args + "\n")

    def _json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _page(self, body: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if dev_reload.enabled():  # lancé par interface/dev_reload.py : la page se recharge toute seule
            if self.path == "/__dev":
                return self._json(HTTPStatus.OK, dev_reload.status(HERE / "index.html"))
            if self.path in ("/", "/index.html"):
                return self._page(dev_reload.page(HERE / "index.html"))
        if self.path == "/api/model":
            return self._json(HTTPStatus.OK, {**self.current_model().info(), "limites": QUOTA.limits(self)})
        if self.path.split("?")[0].endswith("quota.json"):  # IP des visiteurs
            return self.send_error(HTTPStatus.NOT_FOUND)
        if self.path.startswith("/jobs/"):
            self.path = "/_jobs/" + self.path[len("/jobs/"):]
        elif self.path.startswith("/_jobs"):
            return self.send_error(HTTPStatus.NOT_FOUND)
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/analyze":
            return self.send_error(HTTPStatus.NOT_FOUND)
        size = int(self.headers.get("Content-Length") or 0)
        if not size:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "vidéo vide"})
        QUOTA.cleanup()
        try:
            ip = QUOTA.admit(self)
        except Rejected as e:
            self.close_connection = True  # le corps n'est pas lu
            return self._json(e.status, {"error": str(e)})
        try:
            name = Path(self.headers.get("X-Filename") or "video.mp4").name
            job = JOBS / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
            job.mkdir(parents=True)
            video = job / ("video" + (Path(name).suffix.lower() or ".mp4"))
            with open(video, "wb") as fh:
                left = size
                while left:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    fh.write(chunk)
                    left -= len(chunk)
            QUOTA.check_video(ip, video)
            self._json(HTTPStatus.OK, analyze(self.current_model(), video, job))
        except Rejected as e:
            shutil.rmtree(job, ignore_errors=True)
            self._json(e.status, {"error": str(e)})
        except Exception as e:
            QUOTA.refund(ip)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(e).__name__}: {e}"})
        finally:
            QUOTA.release()

def main() -> None:
    p = argparse.ArgumentParser(prog="interface/modele/server.py")
    p.add_argument("--exp", help="expérience à tester (défaut : celle de production.json)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8780)))
    a = p.parse_args()
    print(f"chargement du modèle ({DEVICE})…", flush=True)
    Handler.exp_forced, Handler.prod_mtime = a.exp, _prod_mtime()
    Handler.model = Model(a.exp)
    print(f"modèle {Handler.model.exp_id} prêt — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
