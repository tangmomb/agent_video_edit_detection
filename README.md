# edit_generator

Génère des vidéos annotées pour l'agent d'extraction : clips libres de droits + éléments de montage
aléatoires (mots-clés, questions, sous-titres, bandeaux, images, formes) avec leur vérité terrain.

Sources (`00_edit_generator/sources.py`, choisies selon les clés présentes dans `.env`) :
- vidéos : Pixabay (`PIXABAY_API_KEY`) ;
- photos incrustées : Unsplash (`UNSPLASH_ACCESS_KEY`), sinon images synthétiques.

Les réponses d'API sont mises en cache 24 h dans `data/cache/api/` (exigé par Pixabay, utile pour le
quota Unsplash de 50 req/h).
Les quotas sont suivis par `00_edit_generator/ratelimit.py` (état persisté dans `data/cache/api/`) :
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
edit_generator generate --n 20 --seed 0 --preview            # requêtes aléatoires
edit_generator generate --n 5 --query pasta --orientation portrait
edit_generator generate --local-video clip.mp4 --seed 1      # sans banque de vidéos
edit_generator preview data/videos_samples/s000000                  # planche avec les bbox
```

Sortie : `data/videos_samples/s<seed>/video.mp4` + `annotations.json` (format décrit dans
`00_edit_generator/annotations.py`, avec `bbox_at(el, t)` / `opacity_at(el, t)` pour les animations).

## model_training_pipeline : classifieur « la frame contient un ajout de montage »

```bash
python -m model_training_pipeline frames    # frames à 1 fps + labels (data/videos_samples_frames/index.jsonl)
python -m model_training_pipeline info      # état du dataset
```

Cible binaire `label_any` (au moins un ajout visible sur l'image).
Split train/val/test par rush brut (pas de fuite entre échantillons tirés du même rush).
Métrique principale : rappel à précision fixée (95 % par défaut), seuil choisi sur val et
appliqué tel quel sur test ; AUC et AP en repères (`model_training_pipeline/metrics.py`).

## Agents : pipeline de création du classifieur

```
Orchestrateur ──► Dataset Builder   (génère données + annotations)
      │      ──► Model Researcher  (entraîne, expérimente sur val)
      │      ──► Evaluator         (mesure, analyse les erreurs, test final)
      └──◄── rapports des agents : décide du cycle suivant
```

```bash
python -m agents --budget-usd 1.00 --data-budget 250 --final-tests 2   # appels API payants
```

Organisation :
- `model_training_pipeline/lab.py` : toute la logique métier (données, entraînement, évaluation), sans LLM,
  utilisable et testable seule ;
- `agents/runner.py` : boucle tool use commune sur l'API Mistral (`MISTRAL_API_KEY`), budget
  partagé, journaux. Une réserve (15 % du budget, 0,05 $ minimum) est gardée pour le rapport
  final de l'orchestrateur : les sous-agents s'arrêtent avant, les délégations sont alors
  refusées, et un agent à court de budget ou de tours rédige un dernier rapport sans outils au
  lieu d'être coupé net. Les quatre agents utilisent Mistral Large 4. Chaque rapport de sous-agent
  se termine par une « Vérification automatique » (outils réellement appelés, échecs) écrite par
  le code, pour que l'orchestrateur ne croie pas un rapport inventé ;
- `agents/<agent>.py` : prompt + sous-ensemble d'outils du lab propres à chaque agent ;
  le Dataset Builder génère sans se soucier des splits, le Model Researcher découpe le dataset en
  splits train/val/test nommés et figés (`make_split`, fichiers `data/splits/split_XXX.json`, par
  rush brut, stratifiés par défaut sur type, orientation et clips vides ;
  l'ancien découpage `hash`, non figé, est refusé). Le test est figé au premier split
  (`data/splits/test_set.json`) : les splits suivants ne redécoupent que train/val. Le Dataset
  Builder ne fait qu'ajouter des données ; le Model Researcher peut exclure un clip défectueux
  (`exclude_clips`, `data/excluded.json`), ce qui ne vaut que pour les splits créés ensuite. Seul l'Evaluator mesure sur test.
  Aucun agent ne met de modèle en production : en fin de run, le code (`Lab().select_production()`)
  écrit dans `production.json` l'expérience de meilleur rappel val à précision 95 % sur le plus gros
  split (le plus de vidéos) — départage par AP, AUC puis ms/frame — et l'ajoute au rapport. On peut
  forcer un autre choix avec `Lab().save_model(exp_id, precision, note)` ;
- `agents/orchestrator.py` : délègue via `ask_dataset_builder` / `ask_model_researcher` /
  `ask_evaluator` ; chaque délégation repart d'une conversation vide (l'état vit sur disque).

Journaux : `runs/<NNN>_<horodatage>/pipeline.json` (toutes les étapes de tous les agents, dans
l'ordre, réécrit à chaque étape) + `report.md`. Expériences :
`model_results/` (un dossier `<exp_id>/` par entraînement, `production.json`). Historique des
générations de vidéos : `data/data_requests.jsonl`.
