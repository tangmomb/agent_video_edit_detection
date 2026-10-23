"""Model Researcher : choisit les représentations et les modèles, entraîne, mène les expériences."""

from model_training_pipeline.features import EXTRACTORS
from model_training_pipeline.lab import BALANCE_GROUPS, MODELS, STRATIFY_KEYS, Lab

from .common import CONTEXT, dataset_info_tool, list_experiments_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Model Researcher. Tu constitues les splits train/val/test (make_split) et tu mènes les \
expériences demandées par l'orchestrateur : choix des embeddings figés (CLIP seul, DINOv2 seul ou \
les deux concaténés ; image entière ou quadrants), du classifieur et de ses hyperparamètres, de \
l'information temporelle, et du rééquilibrage de train (balance) quand un type d'ajout ou les \
négatifs sont sur-représentés. Tu ne génères pas de données et tu ne mesures jamais sur test : \
seul l'Evaluator y accède.

Splits : les splits existants sont dans list_experiments ; décide-les à partir de la composition du \
dataset (dataset_info), jamais des erreurs d'un \
modèle. Crée un split quand aucun n'existe, quand le dataset a nettement grossi (clips_hors_split) \
ou quand val ne couvre pas les cas importants (types d'ajout, animations, orientation, thèmes comme \
les négatifs difficiles) ; garde la stratification par défaut (toutes les caractéristiques) sauf \
raison précise. Le test est figé au premier split : les suivants ne redécoupent que train et val, \
et seul exclude_clips retire un clip des splits. Ne refais jamais un split pour améliorer un \
score val : un val choisi pour être facile ne mesure plus rien. \
Vérifie la composition de val et test (dataset_info avec split_id) avant d'entraîner.

Exclusions : exclude_clips retire tout de suite un clip de tous les splits, existants et futurs. \
Ne l'utilise que pour un défaut du clip lui-même (rendu cassé, doublon), jamais parce qu'un modèle \
s'y trompe : les cas où le modèle échoue sont précisément ceux qu'il faut garder, et retirer un \
clip difficile du val fausse la mesure. Justifie chaque exclusion dans ton rapport.

Méthode : consulte l'historique pour ne pas refaire une expérience existante. Formule une hypothèse \
par expérience (champ note : l'hypothèse seule, en une phrase, sans recopier la config ni le split, \
déjà enregistrés) et change un facteur à la fois, sur le même split, pour que les \
résultats soient comparables. Après un nouveau split, ré-entraîne la meilleure config précédente \
dessus pour avoir une référence. Arrête-toi quand le lot demandé est fait ou quand les résultats ne \
bougent plus.

Rapport final (court, en français) : le split utilisé (et pourquoi si tu l'as créé) ; pour chaque \
expérience, son id, l'hypothèse et le résultat (rappel à 95 % de précision, AUC, AP, ms/image) ; le \
meilleur candidat selon toi et pourquoi ; ce que tu essaierais ensuite et les données qui te \
manquent, le cas échéant."""


def make(lab: Lab) -> Agent:
    return Agent(
        name="model_researcher",
        system=SYSTEM,
        max_turns=20,
        tools=[
            dataset_info_tool(lab, "model_researcher"),
            list_experiments_tool(lab),
            Tool(
                name="make_split",
                description="Crée un split train/val/test figé sur toutes les vidéos actuelles (les vidéos "
                            "générées ensuite n'y entrent pas). Les vidéos d'un même rush brut restent du "
                            "même côté. Le test est commun à tous les splits (figé au premier) : seuls train "
                            "et val sont redécoupés. Renvoie l'id du split et la composition de train, val et test.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "note": {"type": "string", "description": "Pourquoi ce split (ce qu'il corrige ou couvre)."},
                        "val_ratio": {"type": "number", "minimum": 0.05, "maximum": 0.4,
                                      "description": "Part des clips en val (défaut 0.2)."},
                        "test_ratio": {"type": "number", "minimum": 0.05, "maximum": 0.4,
                                       "description": "Part des clips en test (défaut 0.2), pour le premier "
                                                      "split seulement : ignoré ensuite (test figé)."},
                        "stratify": {"type": "array", "items": {"type": "string", "enum": list(STRATIFY_KEYS)},
                                     "description": "Caractéristiques à répartir au plus près des proportions "
                                                    "visées dans chaque split : type (d'élément), animation "
                                                    "(d'entrée), orientation, theme, clip_vide. Par défaut toutes ; [] = "
                                                    "tirage purement aléatoire."},
                        "seed": {"type": "integer", "description": "Graine du tirage (défaut 0)."},
                    },
                    "required": ["note"],
                },
                fn=lab.make_split,
            ),
            Tool(
                name="train_model",
                description="Entraîne un classifieur binaire (ajout de montage visible ou non) sur des "
                            "embeddings figés, sur le train du split choisi avec un sous-échantillonnage "
                            "optionnel (balance), et l'évalue sur le val de ce split. Renvoie l'id d'expérience, AUC, AP, "
                            "le rappel au seuil de précision 95 % (avec FP/FN), le coût de calcul "
                            "(ms/image) et l'effet du rééquilibrage. Les embeddings absents du cache sont calculés à la volée (GPU).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "features": {"type": "array", "items": {"type": "string"},
                                     "description": "Liste 'extracteur:vue', extracteurs " + ", ".join(EXTRACTORS)
                                                    + " ; vues full (image entière) ou grid2 (image + 4 quadrants). "
                                                    "Un seul élément = CLIP seul ou DINOv2 seul ; plusieurs = "
                                                    "embeddings concaténés, ex. ['clip_b32:full', "
                                                    "'dinov2_s:full'] pour CLIP + DINOv2."},
                        "model": {"type": "string", "enum": list(MODELS)},
                        "temporal": {"type": "string", "enum": ["none", "diff"],
                                     "description": "diff ajoute e_t - e_(t-1) (changement depuis l'image "
                                                    "précédente)."},
                        "params": {"type": "object",
                                   "description": "logreg: C ; mlp: hidden (liste), alpha ; "
                                                  "hgb: max_iter, lr, max_leaf_nodes ; tous: pca (int)."},
                        "balance": {"type": "object", "additionalProperties": {"type": "number"},
                                    "description": "Sous-échantillonnage de train seulement (val et test "
                                                   "intacts) : {groupe: fraction d'images gardée, 0 à 1}. "
                                                   "Groupes : " + ", ".join(BALANCE_GROUPS) + ". Un type "
                                                   "d'élément = images positives dont tous les ajouts visibles "
                                                   "sont de ce type ; negatif = images sans ajout ; clip_vide "
                                                   "= images des clips sans aucun ajout. Ex. "
                                                   "{'subtitle': 0.5, 'negatif': 0.7}. Le dataset n'est pas "
                                                   "modifié."},
                        "note": {"type": "string",
                                 "description": "Hypothèse testée, en une phrase, sans recopier la config."},
                        "split_id": {"type": "string",
                                     "description": "Split sur lequel entraîner (train) et mesurer (val) : "
                                                    "un id renvoyé par make_split (voir list_experiments)."},
                    },
                    "required": ["features", "split_id"],
                },
                fn=lab.train_model,
            ),
            Tool(
                name="exclude_clips",
                description="Retire des clips de tous les splits, présents et futurs, sans supprimer de "
                            "fichier. Réservé aux clips défectueux ou dupliqués ; s'applique tout de suite aux "
                            "splits existants.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "clips": {"type": "object", "additionalProperties": {"type": "string"},
                                  "description": "{sample_id: raison du défaut}"},
                    },
                    "required": ["clips"],
                },
                fn=lab.exclude_clips,
            ),
        ],
    )
