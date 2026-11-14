"""Entraînement de train_model sur un GPU Modal : allumé par l'appel, éteint 20 s après le
dernier, facturé à la seconde. Sur le VPS, la matrice d'embeddings et les copies de scikit-learn
grandissaient avec le dataset jusqu'à l'OOM (run 025 tué à 5 Go ; run 028 vers 4,6 à 5 Go).

Deux fonctions, pour que le GPU n'attende jamais les lectures :
- pack (CPU seul, environ 20 fois moins cher à l'heure qu'un GPU) : lit en parallèle les embeddings
  des images demandées dans le volume agent-features (copie des caches du VPS, tenue à jour par
  feature_volume.py) et les écrit en un seul fichier du volume, gardé : les train_model suivants
  qui demandent les mêmes images et embeddings le réutilisent. Une première version lisait les
  fichiers un par un sur le GPU : 35 min de T4 à 0 % pour 17 000 fichiers, sans finir ;
- train (GPU) : relit ce fichier d'un coup et entraîne (train_torch.py).

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
PACKS = "packs"  # matrices assemblées, dans le volume
KEEP_PACKS = 12  # les plus récemment utilisées ; les autres sont effacées
READ_THREADS = 64

TRAIN_TORCH = Path(__file__).with_name("train_torch.py")

# Mêmes versions que le VPS (torch, numpy) ; pas de scikit-learn : le VPS reconstruit les modèles.
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "numpy==2.5.2")
         .add_local_file(TRAIN_TORCH, "/root/train_torch.py", copy=True)
         .env({"PYTHONPATH": "/root"}))

app = modal.App(APP, image=image)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


# Modal facture la mémoire et les CPU réservés (ou utilisés, si plus), en plus du GPU, et un conteneur
# reste payé pendant scaledown_window après son dernier appel. Run 029 (14/11) : 1,42 $ d'entraînement
# pour 24 min de calcul, le conteneur GPU (4 CPU, 32 Go) ne s'éteignant presque jamais entre deux
# train_model (un par minute environ). D'où des réserves au plus juste et un arrêt rapide.
# Plus grosse matrice à ce jour : 21 000 images x 18 000 dims, 1,5 Go en float32.
IDLE_S = 20


@app.function(cpu=4,  # lectures et décompression des npz en parallèle
              memory=8192, timeout=3600, scaledown_window=IDLE_S, max_containers=1, retries=1,
              volumes={ROOT: volume})
def pack(payload: dict) -> dict:
    """Assemble la matrice des images demandées dans <volume>/packs/<key>.npy (déjà là : rien à
    faire). Renvoie {"pack": key, ...} ou {"missing": [(dossier, vidéo)...]} si le volume n'a pas
    encore ces images."""
    import numpy as np
    import train_torch

    volume.reload()  # les fichiers envoyés juste avant par le VPS
    packs = Path(ROOT) / PACKS
    path = packs / f"{payload['key']}.npy"
    if path.exists():
        path.touch()  # récemment utilisée : gardée par le ménage ci-dessous
        volume.commit()
        return {"pack": payload["key"], "reutilise": True}
    t0 = time.time()
    X, missing = train_torch.load_matrix(Path(ROOT) / payload["namespace"], payload["specs"],
                                         payload["sids"], payload["frames"], threads=READ_THREADS)
    if missing:
        return {"missing": missing}
    t1 = time.time()
    packs.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, X)
    tmp.replace(path)
    for old in sorted(packs.glob("*.npy"), key=lambda p: p.stat().st_mtime)[:-KEEP_PACKS]:
        old.unlink(missing_ok=True)
    volume.commit()
    return {"pack": payload["key"], "reutilise": False, "shape": list(X.shape),
            "lecture_s": round(t1 - t0, 1), "ecriture_s": round(time.time() - t1, 1)}


# Plusieurs GPU acceptés : un L4 seul peut rester des minutes en attente de place.
@app.function(gpu=["L4", "T4", "A10"], cpu=2,  # le calcul est sur le GPU
              memory=12288,  # la matrice une fois en RAM avant d'aller sur le GPU
              timeout=3600, scaledown_window=IDLE_S,
              max_containers=1,  # plafond de dépense : jamais deux GPU en même temps
              retries=1,  # GPU repris par Modal en plein calcul
              volumes={ROOT: volume})
def train(payload: dict) -> dict:
    """Entraîne un modèle par pli (train_torch.run) sur la matrice assemblée par pack."""
    import numpy as np
    import torch
    import train_torch

    volume.reload()
    t0 = time.time()
    X = np.load(Path(ROOT) / PACKS / f"{payload['pack']}.npy")
    t1 = time.time()
    out = train_torch.run(X, payload["y"], payload["plan"])
    out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    out["timing"] = {"chargement_s": round(t1 - t0, 1), "entrainement_s": round(time.time() - t1, 1)}
    return out
