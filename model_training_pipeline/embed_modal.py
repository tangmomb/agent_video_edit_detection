"""Calcul des embeddings sur un GPU Modal : allumé par l'appel, éteint une minute après le dernier,
facturé à la seconde. Sur le CPU du VPS, un gros extracteur bloquait un run pendant des heures
(run 017 : environ 16 h sur 17 h 30).

Déploiement, à refaire si encoders.py ou ce fichier change :
    modal deploy model_training_pipeline/embed_modal.py
Utilisé par get_features (features.py) quand EMBED_GPU=modal dans .env.
"""

import io
from pathlib import Path

import modal

APP = "agent-embeddings"
BATCH_SIZE = 256  # images envoyées au modèle d'un coup (5 par frame en grid2)

ENCODERS = Path(__file__).with_name("encoders.py")


def _download_models():
    """Pendant la construction de l'image : les modèles sont dedans, pas retéléchargés à chaque démarrage."""
    import encoders

    for name in encoders.EXTRACTORS:
        encoders._encoder(name, "cpu")
        encoders._encoders.clear()


# Mêmes versions que le VPS, pour que décodage et prétraitement ne changent pas les embeddings.
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.20.2", "pillow==13.0.0", "numpy==2.5.2")
         .add_local_file(ENCODERS, "/root/encoders.py", copy=True)
         .env({"PYTHONPATH": "/root"})
         .run_function(_download_models))

app = modal.App(APP, image=image)


# Plusieurs GPU acceptés : un L4 seul peut rester des minutes en attente de place.
@app.function(gpu=["L4", "T4", "A10"], cpu=4, memory=16384, timeout=600, scaledown_window=60,
              max_containers=1,  # plafond de dépense : jamais deux GPU en même temps
              retries=2)  # GPU repris par Modal en plein calcul
def embed(jpegs: list[bytes], extractor: str, view: str) -> bytes:
    """Les embeddings float32 de ces images, à la suite ; l'appelant connaît le nombre d'images."""
    import encoders
    import numpy as np
    import torch

    with torch.inference_mode():
        emb = encoders.embed_images([io.BytesIO(b) for b in jpegs], extractor, view, BATCH_SIZE)
    return emb.astype(np.float32).tobytes()
