"""Entraînement de train_model sur un GPU Modal : allumé par l'appel, éteint une minute après le
dernier, facturé à la seconde. Sur le VPS, la matrice d'embeddings et les copies de scikit-learn
grandissaient avec le dataset jusqu'à l'OOM (run 025 tué à 5 Go ; run 028 vers 4,6 à 5 Go).

Les embeddings sont lus dans le volume Modal agent-features (copie des caches du VPS, tenue à jour
par feature_volume.py avant chaque appel) ; le calcul est celui de train_torch.py.

Déploiement, à refaire si ce fichier ou train_torch.py change (fait par le workflow de déploiement) :
    modal deploy model_training_pipeline/train_modal.py
Utilisé par Lab.train_model (remote_train.py) quand TRAIN_GPU=modal dans .env.
"""

import time
from pathlib import Path

import modal

APP = "agent-training"
VOLUME = "agent-features"
ROOT = "/features"

TRAIN_TORCH = Path(__file__).with_name("train_torch.py")

# Mêmes versions que le VPS (torch, numpy) ; pas de scikit-learn : le VPS reconstruit les modèles.
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "numpy==2.5.2")
         .add_local_file(TRAIN_TORCH, "/root/train_torch.py", copy=True)
         .env({"PYTHONPATH": "/root"}))

app = modal.App(APP, image=image)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


# Plusieurs GPU acceptés : un L4 seul peut rester des minutes en attente de place.
@app.function(gpu=["L4", "T4", "A10"], cpu=4, memory=32768, timeout=3600, scaledown_window=60,
              max_containers=1,  # plafond de dépense : jamais deux GPU en même temps
              retries=2,  # GPU repris par Modal en plein calcul
              volumes={ROOT: volume})
def train(payload: dict) -> dict:
    """Charge les embeddings des images demandées, entraîne un modèle par pli (train_torch.run).
    Renvoie {"missing": [(dossier, vidéo)...]} si le volume n'a pas encore ces images."""
    import train_torch

    volume.reload()  # les fichiers envoyés juste avant par le VPS
    t0 = time.time()
    X, missing = train_torch.load_matrix(Path(ROOT) / payload["namespace"], payload["specs"],
                                         payload["sids"], payload["frames"])
    if missing:
        return {"missing": missing}
    t1 = time.time()
    out = train_torch.run(X, payload["y"], payload["plan"])
    out["timing"] = {"chargement_s": round(t1 - t0, 1), "entrainement_s": round(time.time() - t1, 1)}
    return out
