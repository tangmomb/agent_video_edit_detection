# Détecteur d'ajouts de montage

Classifieur d'images : « cette image de vidéo contient-elle un ajout de montage (habillage, texte,
sous-titres, image incrustée, logo…) ou est-elle brute de la caméra ? ». Il est entraîné sur de
vraies vidéos YouTube, annotées image par image par trois LLM de vision.

## video_source : vraies vidéos et annotation

- `00_video_source/youtube.py` : recherche et téléchargement YouTube avec yt-dlp (aucune clé). Une
  vidéo n'est prise qu'une fois, 5 min au plus, image seule en ≤ 360p (images réduites à 640 px de toute façon) ; portrait = Shorts. Sortie :
  `data/videos_samples/yt_<id>/video.*` + `meta.json` (titre, chaîne, url, requête, dimensions).
  `YT_PROXY` (`.env`) fait passer tout le trafic YouTube par un relais, pour une machine dont
  YouTube bloque l'IP (VPS Contabo : tunnel SSH vers le VPS Infomaniak, `socks5://127.0.0.1:1080`).
- `00_video_source/annotators.py` : une requête par image, réponse `{"ajout": true|false}`. Le
  dataset est annoté par `gpt-6-luna`, `gpt-5.6-luna` et `zai-glm-5-3-flash` (les meilleurs du banc
  `utils/llm_bench`) : une image n'a d'étiquette que s'ils sont **tous d'accord**, sinon elle est
  écartée (ni entraînée ni mesurée), sans revue manuelle. Votes dans `votes.json` à côté de la vidéo.
  Effort de raisonnement `medium` pour les GPT, aucun effort précisé pour `zai-glm-5-3-flash` (son
  API n'accepte ni `medium` ni `none`). L'annotation passe par les API batch d'OpenAI et de Mistral (-50 %),
  suivies dans `data/batches.json`. Une image presque identique à la dernière image annotée de sa
  vidéo (cosinus CLIP > 0.87) n'est pas annotée : environ 70 % des images en moins à payer, pour
  20 % des apparitions ou disparitions d'ajout manquées (`frames.to_annotate`, seuil réglable à l'œil
  dans `utils/skip_tester`).

## Installation

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .   # Linux : .venv/bin/python
cp .env.example .env                        # puis renseigner les clés
```

ffmpeg est fourni par `imageio-ffmpeg` (aucune installation système).

## Utilisation

```bash
video_source collect --query "recette pâtes rapide" --n 5                # téléchargement seul (gratuit)
video_source collect --query "vlog voyage" --n 3 --orientation portrait
python -m model_training_pipeline batch estimate                       # requêtes à envoyer, coût estimé (gratuit)
python -m model_training_pipeline batch submit --max-usd 0.50          # vote des 3 LLM en batch (payant, -50 %)
python -m model_training_pipeline batch status
python -m model_training_pipeline batch collect                        # votes des batchs terminés -> index
python -m model_training_pipeline annotate --max-usd 0.50              # variante en appels directs (plein tarif)
video_source preview data/videos_samples/yt_<id>                        # planche avec les étiquettes
```

## model_training_pipeline : classifieur « la frame contient un ajout de montage »

```bash
python -m model_training_pipeline frames    # frames à 1 fps + index (data/videos_samples_frames/index.jsonl), sans appel payant
python -m model_training_pipeline info      # état du dataset
python -m model_training_pipeline features  # précalcule les embeddings (tous extracteurs et vues, GPU), sans appel payant
python -m model_training_pipeline exclude <sample_id> "<raison>"  # retire une vidéo défectueuse des splits suivants
```

Cible binaire `label_any` (au moins un ajout visible sur l'image), `null` sans accord des annotateurs.
Split dev/test par chaîne YouTube (une chaîne garde son habillage d'une vidéo à l'autre).
Métrique principale : rappel à précision fixée (95 % par défaut) sur val, une validation croisée
par chaîne sur dev (5 plis : chaque image est prédite par un modèle qui n'a vu aucune vidéo de sa
chaîne). Le seuil y est choisi prudemment (centile 80 des seuils de 200 tirages de chaînes avec
remise), puis appliqué tel quel sur test à la moyenne des modèles de plis ; AUC et AP en repères
(`model_training_pipeline/metrics.py`). Un val fixe ne comptait que 5 à 13 chaînes : son seuil ne
tenait pas la précision sur d'autres chaînes (banc b1 : 0.53 à 1.0 pour 0.95 visé).

## Agents : pipeline de création du classifieur

```
Orchestrateur ──► Dataset Builder   (collecte vidéos YouTube + annotation par 3 LLM)
      │      ──► Model Researcher  (entraîne, expérimente en validation croisée)
      │      ──► Evaluator         (mesure, analyse les erreurs, test final)
      └──◄── rapports des agents : décide du cycle suivant
```

```bash
python -m agents --budget-usd 1.00 --annotation-usd 2.00 --data-budget 250 --final-tests 2   # appels API payants
```

Banc d'agents (`agents/bench.py`) : compare des modèles LLM, le même pour les quatre agents, sur un
instantané figé du dataset dans `bench/<nom>/`, sans toucher aux vraies données. Une part des
chaînes est cachée aux agents ; le modèle de production de chaque run y est mesuré : AP, rappel
et précision au seuil choisi par le run (la précision visée tient-elle ?) et rappel idéal à 95 %
(meilleur seuil possible : qualité du modèle sans l'erreur de seuil).

```bash
python -m agents.bench snapshot --name b1                  # instantané + jeu caché (gratuit)
python -m agents.bench run --name b1 --runs 5 --budget 1.0 # 6 modèles x 5 runs (appels API payants)
python -m agents.bench score --name b1                     # tableau bench/b1/results.md (gratuit)
```

Organisation :
- `model_training_pipeline/lab.py` : toute la logique métier (données, entraînement, évaluation), sans LLM,
  utilisable et testable seule ;
- `agents/runner.py` : boucle tool use commune via LiteLLM (Mistral, OpenAI, Anthropic, Google ;
  clés du `.env`), budget
  partagé, journaux. Une réserve (15 % du budget, 0,05 $ minimum) est gardée pour le rapport
  final de l'orchestrateur : les sous-agents s'arrêtent avant, les délégations sont alors
  refusées, et un agent à court de budget ou de tours rédige un dernier rapport sans outils au
  lieu d'être coupé net. Les quatre agents utilisent Mistral Large 4 (choisi d'après le banc `agents.bench` b2), ou le
  modèle donné par `--agent-model` (même modèle pour les quatre, ex. `openai/gpt-5.6-luna`). Chaque rapport de sous-agent
  se termine par une « Vérification automatique » (outils réellement appelés, échecs) écrite par
  le code, pour que l'orchestrateur ne croie pas un rapport inventé ;
- `agents/<agent>.py` : prompt + sous-ensemble d'outils du lab propres à chaque agent ;
  le Dataset Builder collecte sans se soucier des splits (`generate_data` : recherche YouTube,
  téléchargement, annotation en mode batch dans la limite de `--annotation-usd`, attente des résultats
  sans limite, dans le même appel), le Model Researcher découpe le dataset en
  splits dev/test nommés et figés (`make_split`, fichiers `data/splits/split_XXX.json`, par
  chaîne YouTube, stratifiés par défaut sur orientation et vidéos sans ajout, comme les plis de
  dev ; l'ancien découpage `hash`, non figé, est refusé ; les anciens splits train/val/test se
  lisent comme dev/test). Le test est figé au premier split (`data/splits/test_set.json`) : les
  splits suivants mettent tout le reste en dev. Le Dataset
  Builder ne fait qu'ajouter des données ; un clip défectueux (vidéo cassée, doublon) s'exclut à la
  main (`python -m model_training_pipeline exclude <sample_id> "<raison>"`, `data/excluded.json`),
  ce qui ne vaut que pour les splits créés ensuite. Seul l'Evaluator mesure sur test.
  Ses exemples d'erreurs (`show_examples`) sont les plus franches, une vidéo à tour de rôle. Vues d'embeddings : `full`, `grid2` (image + 4 quadrants) et `bands` (image + tiers bas
  + tiers haut, pour les bandeaux et sous-titres).
  Aucun agent ne met de modèle en production : en fin de run, le code (`Lab().select_production()`)
  écrit dans `production.json`, sur le plus gros split (le plus de clips), l'expérience de meilleure
  AP par chaîne sur val (moyenne des AP de chaque chaîne) parmi celles qui ont réussi leur test final
  (précision visée tenue sur test) — départage par rappel val à 95 %, AP, AUC puis temps de calcul.
  Banc b2 rejoué : le rappel val gardait souvent CLIP seul, qui ne tient pas sur des chaînes
  nouvelles ; l'AP par chaîne fait passer le rappel idéal caché du modèle choisi de 0.55 à 0.66. Sans modèle testé avec succès, il prend le
  meilleur sur val et le marque « non vérifié sur test » dans le rapport. On peut
  forcer un autre choix avec `Lab().save_model(exp_id, precision, note)` ;
- `agents/orchestrator.py` : délègue via `ask_dataset_builder` / `ask_model_researcher` /
  `ask_evaluator` ; chaque délégation repart d'une conversation vide (l'état vit sur disque).

Journaux : `runs/<NNN>_<horodatage>/pipeline.json` (toutes les étapes de tous les agents, dans
l'ordre, réécrit à chaque étape) + `report.md`. Expériences :
`model_results/` (un dossier `<exp_id>/` par entraînement, `production.json`). Historique des
collectes de vidéos : `data/data_requests.jsonl`.
