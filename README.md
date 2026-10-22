# editgen

Génère des vidéos annotées pour l'agent d'extraction : clips libres de droits + éléments de montage
aléatoires (mots-clés, questions, sous-titres, bandeaux, images, formes) avec leur vérité terrain.

Sources (`editgen/sources.py`, choisies selon les clés présentes dans `.env`) :
- vidéos : Pixabay (`PIXABAY_API_KEY`) ;
- photos incrustées : Unsplash (`UNSPLASH_ACCESS_KEY`), sinon images synthétiques.

Les réponses d'API sont mises en cache 24 h dans `data/cache/api/` (exigé par Pixabay, utile pour le
quota Unsplash de 50 req/h).
Les quotas sont suivis par `editgen/ratelimit.py` (état persisté dans `data/cache/api/`) :
Pixabay attend jusqu'à 90 s puis réessaie ; Unsplash n'attend pas (photos en cache ou
synthétiques). Une génération s'interrompt après 3 échecs de quota d'affilée, et seules les
vidéos réussies consomment le budget. `dataset_info` expose l'état (`quotas_api`).

## Installation

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .   # Linux : .venv/bin/python
cp .env.example .env                        # puis renseigner les clés
```

ffmpeg est fourni par `imageio-ffmpeg` (aucune installation système).
Sur un serveur sans polices, déposer des `.ttf` dans `assets/fonts/`.

## Utilisation

```bash
editgen generate --n 20 --seed 0 --preview            # requêtes aléatoires
editgen generate --n 5 --query pasta --orientation portrait
editgen generate --local-video clip.mp4 --seed 1      # sans banque de vidéos
editgen preview data/samples/s000000                  # planche avec les bbox
```

Sortie : `data/samples/s<seed>/video.mp4` + `annotations.json` (format décrit dans
`editgen/annotations.py`, avec `bbox_at(el, t)` / `opacity_at(el, t)` pour les animations).

## selector : classifieur « la frame contient un ajout de montage »

```bash
python -m selector frames    # frames à 1 fps + labels (data/frames/index.jsonl)
python -m selector info      # état du dataset
```

Cible binaire `label_any` (au moins un ajout pleinement visible), frames en fondu exclues.
Split train/val/test par vidéo *source* (pas de fuite entre échantillons tirés du même clip).
Métrique principale : rappel à précision fixée (95 % par défaut), seuil choisi sur val et
appliqué tel quel sur test ; AUC et AP en repères (`selector/metrics.py`).

## Agents : pipeline de création du classifieur

```
Orchestrateur ──► Dataset Builder   (génère données + annotations)
      │      ──► Model Researcher  (entraîne, expérimente sur val)
      │      ──► Evaluator         (mesure, analyse les erreurs, test, production)
      └──◄── rapports des agents : décide du cycle suivant
```

```bash
python -m agents --budget-usd 1.00 --data-budget 200 --final-tests 2   # appels API payants
```

Organisation :
- `selector/lab.py` : toute la logique métier (données, entraînement, évaluation), sans LLM,
  utilisable et testable seule ;
- `agents/runner.py` : boucle tool use commune sur l'API Mistral (`MISTRAL_API_KEY`), budget
  partagé, journaux. Les quatre agents utilisent Mistral Large 4. Chaque rapport de sous-agent
  se termine par une « Vérification automatique » (outils réellement appelés, échecs) écrite par
  le code, pour que l'orchestrateur ne croie pas un rapport inventé ;
- `agents/<agent>.py` : prompt + sous-ensemble d'outils du lab propres à chaque agent ;
  seul l'Evaluator accède au split test et à la mise en production ;
- `agents/orchestrator.py` : délègue via `ask_dataset_builder` / `ask_model_researcher` /
  `ask_evaluator` ; chaque délégation repart d'une conversation vide (l'état vit sur disque).

Journaux : `data/runs/<horodatage>/pipeline.json` (toutes les étapes de tous les agents, dans
l'ordre, réécrit à chaque étape) + `report.md`. Expériences :
`data/experiments/` (`experiments.jsonl`, `data_requests.jsonl`, `production.json`).
