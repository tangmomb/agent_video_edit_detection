"""Interface de test du modèle : on envoie une vidéo, on voit les frames où il détecte un ajout.

    .venv/Scripts/python.exe interface/server.py [--exp exp_037] [--port 8780]

Sans --exp, utilise le modèle désigné dans model_results/production.json (et son seuil). La vidéo
est traitée exactement comme à l'entraînement : frames à FPS par seconde (frames.extract), mêmes
embeddings (features), même composante temporelle (lab._X), même classifieur.
Fichiers de travail dans interface/_jobs/<id>/ (frames + résultat), effaçables à tout moment.

Une vidéo peut ensuite être annotée par les trois LLM annotateurs du dataset (bouton de la page,
appels payants, plafonnés par vidéo ; une image n'a d'étiquette que s'ils sont d'accord) puis
corrigée à la main : elle est alors rangée dans data/real_videos/<id>/ (voir annotate.py)
et la page compare le modèle à ces étiquettes.
"""

import argparse
import json
import os
import pickle
import sys
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_training_pipeline import metrics  # noqa: E402
from model_training_pipeline.features import _encoder, _views  # noqa: E402
from model_training_pipeline.frames import FPS, extract  # noqa: E402
from model_training_pipeline.lab import EXP_DIR  # noqa: E402
from utils import dev_reload  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import annotate  # noqa: E402
JOBS = HERE / "_jobs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_gpu = Lock()  # une analyse à la fois sur le GPU


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
            preds = json.loads((d / "val_preds.json").read_text(encoding="utf-8"))
            self.precision = metrics.DEFAULT_PRECISION
            op = metrics.recall_at_precision(np.array(preds["y"]), np.array(preds["probs"]), self.precision)
            self.threshold = op["threshold"]
        for spec in self.config["features"]:  # charge les encodeurs au démarrage, pas à la 1re vidéo
            _encoder(spec.partition(":")[0], DEVICE)

    def info(self) -> dict:
        ft = (self.final_test or {}).get("test")
        return {"experiment_id": self.exp_id, "features": self.config["features"],
                "model": self.config["model"], "temporal": self.config["temporal"],
                "split_id": self.config.get("split_id"), "threshold": self.threshold,
                "precision_visee": self.precision, "device": DEVICE,
                "test": {k: ft[k] for k in ("precision", "recall")} if ft else None}

    def score(self, frames: list[Path]) -> np.ndarray:
        mats = []
        with torch.inference_mode(), torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            for spec in self.config["features"]:
                ext, _, view = spec.partition(":")
                view = view or "full"
                encode, n_views = _encoder(ext, DEVICE), 5 if view == "grid2" else 1
                out = []
                for i in range(0, len(frames), max(1, 32 // n_views)):
                    imgs = [v for f in frames[i:i + max(1, 32 // n_views)]
                            for v in _views(Image.open(f).convert("RGB"), view)]
                    emb = encode(imgs).float().cpu().numpy()
                    out.append(emb.reshape(-1, n_views * emb.shape[1]))
                mats.append(np.concatenate(out))
        X = np.concatenate(mats, axis=1)
        if self.config["temporal"] == "diff":  # comme lab._X : la 1re frame n'a pas de précédente (0)
            X_prev = np.vstack([np.zeros_like(X[:1]), X[:-1]])
            X = np.concatenate([X, X - X_prev], axis=1)
        return self.clf.predict_proba(X)[:, 1]


def analyze(model: Model, video: Path, job: Path) -> dict:
    t0 = time.perf_counter()
    frames = extract(video, job / "frames", FPS, long_side=640)  # mêmes réglages que frames.build
    if not frames:
        raise ValueError("aucune frame extraite : fichier vidéo illisible ?")
    with _gpu:
        probs = model.score(frames)
    result = {"job": job.name, "fps": FPS, "model": model.info(),
              "duration_s": round(time.perf_counter() - t0, 2),
              "frames": [{"t": round(i / FPS, 3), "prob": round(float(p), 5),
                          "url": f"/jobs/{job.name}/frames/{f.name}"}
                         for i, (f, p) in enumerate(zip(frames, probs))]}
    (job / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result


class Handler(SimpleHTTPRequestHandler):
    model: Model

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
        if dev_reload.enabled():  # lancé par utils/dev_reload.py : la page se recharge toute seule
            if self.path == "/__dev":
                return self._json(HTTPStatus.OK, dev_reload.status(HERE / "index.html"))
            if self.path in ("/", "/index.html"):
                return self._page(dev_reload.page(HERE / "index.html"))
        if self.path == "/api/model":
            return self._json(HTTPStatus.OK, self.model.info())
        if self.path.startswith("/jobs/"):
            self.path = "/_jobs/" + self.path[len("/jobs/"):]
        elif self.path.startswith("/_jobs"):
            return self.send_error(HTTPStatus.NOT_FOUND)
        return super().do_GET()

    def _body(self) -> dict:
        size = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(size) or b"{}")

    def _annotate(self):
        """Étiquettes LLM de la vidéo d'un job (appels payants aux trois annotateurs la première fois seulement)."""
        job = JOBS / Path(self._body().get("job", "")).name
        if not (job / "result.json").exists():
            return self._json(HTTPStatus.NOT_FOUND, {"error": "analyse introuvable"})
        labels = annotate.load(job.name)
        if labels is None:
            result = json.loads((job / "result.json").read_text(encoding="utf-8"))
            annotate.save(job, result, annotate.annotate_video(result, job))
            labels = annotate.load(job.name)
        return self._json(HTTPStatus.OK, labels)

    def _label(self):
        b = self._body()
        return self._json(HTTPStatus.OK, annotate.set_manual(b["job"], int(b["i"]), b.get("value")))

    def do_POST(self):
        routes = {"/api/annotate": self._annotate, "/api/label": self._label}
        if self.path in routes:
            try:
                return routes[self.path]()
            except Exception as e:
                return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(e).__name__}: {e}"})
        if self.path != "/api/analyze":
            return self.send_error(HTTPStatus.NOT_FOUND)
        size = int(self.headers.get("Content-Length") or 0)
        if not size:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "vidéo vide"})
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
        try:
            self._json(HTTPStatus.OK, analyze(self.model, video, job))
        except Exception as e:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(e).__name__}: {e}"})


def main() -> None:
    p = argparse.ArgumentParser(prog="interface/server.py")
    p.add_argument("--exp", help="expérience à tester (défaut : celle de production.json)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8780)))
    a = p.parse_args()
    print(f"chargement du modèle ({DEVICE})…", flush=True)
    Handler.model = Model(a.exp)
    print(f"modèle {Handler.model.exp_id} prêt — http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
