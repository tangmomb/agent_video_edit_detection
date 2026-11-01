"""Model Researcher : choisit les représentations et les modèles, entraîne, mène les expériences."""

from model_training_pipeline.features import EXTRACTORS, VIEWS
from model_training_pipeline.lab import BALANCE_GROUPS, MODELS, STRATIFY_KEYS, Lab

from .common import CONTEXT, analyze_errors_tool, dataset_info_tool, list_experiments_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Model Researcher. Tu constitues les splits dev/test (make_split) et tu mènes les \
expériences demandées par l'orchestrateur : choix des embeddings figés (CLIP seul, DINOv2 seul ou \
les deux concaténés ; image entière ou quadrants), du classifieur et de ses hyperparamètres, du \
rééchantillonnage des images d'entraînement (balance : retirer ou dupliquer des \
images d'un groupe) et de la pondération de la perte (weights : faire peser davantage un groupe \
difficile, par exemple les images brutes que le modèle prend pour des ajouts). Le seuil est choisi par le code sur \
val (meilleur rappel à 95 % de précision, choisi prudemment) : pondérer toutes les images positives ou négatives revient \
surtout à le déplacer ; vise plutôt un groupe précis. Tu ne collectes pas de données et tu ne mesures jamais sur test : \
seul l'Evaluator y accède.

Splits : list_experiments donne le split le plus récent, le seul sur lequel tu entraînes, et une \
ligne par ancien split ayant des expériences ; décide-les à partir de la composition du \
dataset (dataset_info), jamais des erreurs d'un \
modèle. Crée un split quand aucun n'existe ou quand le dataset a nettement grossi \
(clips_hors_split) ; garde la stratification par défaut (toutes les caractéristiques) sauf \
raison précise. Chaque split tire son propre test : les tests finaux de deux splits ne se \
comparent donc pas. Ne refais jamais \
un split pour améliorer un score val : les plis de validation suivent la graine du split, et un \
découpage choisi pour être facile ne mesure plus rien. \
Vérifie la composition de dev et test du split le plus récent (dataset_info) avant d'entraîner.

Méthode : consulte l'historique pour ne pas refaire une expérience existante. Formule une hypothèse \
par expérience (champ note : l'hypothèse seule, en une phrase, sans recopier la config ni le split, \
déjà enregistrés) et change un facteur à la fois pour que les résultats soient comparables. Après un nouveau split, ré-entraîne la meilleure config précédente \
dessus pour avoir une référence. Quand une expérience vise un groupe d'erreurs (pondération, \
rééchantillonnage, vue d'embeddings), compare son analyze_errors à celui de la référence : le \
changement ne vaut que si les erreurs de ce groupe baissent sans que les autres montent, pas \
seulement le score global. Tu ne vois pas les images : pour savoir quels ajouts tu rates (logo, \
sous-titres, bandeau…), demande analyze_errors avec analyse_visuelle, une fois sur ta référence \
plutôt qu'à chaque expérience (payant). \
Arrête-toi quand le lot demandé est fait ou quand les résultats ne bougent plus.

Rapport final (court, en français) : le split utilisé (et pourquoi si tu l'as créé) ; pour chaque \
expérience, son id, l'hypothèse et le résultat (AP par chaîne, rappel à 95 % de précision, AP, et \
l'effet sur le groupe visé) ; le meilleur candidat selon toi (d'abord sur l'AP par chaîne) et pourquoi ; ce que tu essaierais ensuite et les données qui te \
manquent, le cas échéant. Ne compare que des expériences du même split."""


def make(lab: Lab, session=None) -> Agent:
    return Agent(
        name="model_researcher",
        system=SYSTEM,
        max_turns=20,
        tools=[
            dataset_info_tool(lab, "model_researcher"),
            analyze_errors_tool(lab, session),
            list_experiments_tool(lab),
            Tool(
                name="make_split",
                description="Crée un split dev/test, qui ne change plus, sur toutes les vidéos actuelles (les vidéos "
                            "collectées ensuite n'y entrent pas). Les vidéos d'une même chaîne YouTube restent "
                            "du même côté. Chaque split tire son propre test (150 chaînes au plus) ; le reste va "
                            "en dev, découpé par chaîne en plis de validation croisée. Renvoie l'id "
                            "du split, le nombre de plis et la composition de dev et test.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "note": {"type": "string", "description": "Pourquoi ce split (ce qu'il corrige ou couvre)."},
                        "test_ratio": {"type": "number", "minimum": 0.05, "maximum": 0.4,
                                       "description": "Part des chaînes en test (défaut 0.2)."},
                        "stratify": {"type": "array", "items": {"type": "string", "enum": list(STRATIFY_KEYS)},
                                     "description": "Caractéristiques à répartir au plus près des proportions "
                                                    "visées entre dev et test et entre les plis "
                                                    "de dev : orientation, clip_vide (vidéo sans aucune image "
                                                    "avec ajout). Par défaut toutes ; [] = tirage purement "
                                                    "aléatoire."},
                        "seed": {"type": "integer", "description": "Graine du tirage du test et des plis (défaut 0)."},
                    },
                    "required": ["note"],
                },
                fn=lab.make_split,
            ),
            Tool(
                name="train_model",
                description="Entraîne un classifieur binaire (ajout de montage visible ou non) sur des "
                            "embeddings figés, avec un rééchantillonnage (balance) et une pondération "
                            "(weights) optionnels des images d'entraînement, et l'évalue par validation "
                            "croisée par chaîne sur le dev du split le plus récent (val) ; le modèle gardé est la moyenne des modèles de plis. "
                            "Renvoie l'id d'expérience, AUC, AP, AP par chaîne, le rappel au seuil de précision 95 % "
                            "(avec FP/FN) et l'effet du rééchantillonnage et de la pondération. Les embeddings absents du cache sont calculés à la volée (GPU).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "embeddings": {"type": "array",
                                     "items": {"type": "string",
                                               "enum": [f"{e}:{v}" for e in EXTRACTORS for v in VIEWS]},
                                     "description": "Liste 'extracteur:vue' (valeurs de l'enum uniquement), "
                                                    "extracteurs " + ", ".join(EXTRACTORS)
                                                    + " ; vues full (image entière), grid2 (image + 4 quadrants : "
                                                    "logos, petits overlays ; chaque quadrant contient un coin, il "
                                                    "n'y a pas d'autre vue de coins) ou bands (image + tiers bas + "
                                                    "tiers haut : bandeaux de nom, sous-titres, titres). "
                                                    "Un seul élément = CLIP seul ou DINOv2 seul ; plusieurs = "
                                                    "embeddings concaténés, ex. ['clip_b32:full', "
                                                    "'dinov2_s:full'] pour CLIP + DINOv2."},
                        "model": {"type": "string", "enum": list(MODELS),
                                  "description": "logreg (régression logistique) ou mlp (petit réseau de "
                                                 "neurones) : quelques secondes chacun."},
                        "params": {"type": "object",
                                   "description": "logreg: C ; mlp: hidden (liste), alpha ; "
                                                  "tous: pca (int)."},
                        "balance": {"type": "object", "additionalProperties": {"type": "number"},
                                    "description": "Rééchantillonnage des images d'entraînement seulement "
                                                   "(images mesurées et test intacts) : {groupe: facteur, 0 à 10}. Sous 1, on garde "
                                                   "cette fraction des images ; au-dessus, on les duplique "
                                                   "(3 = trois exemplaires). Groupes : "
                                                   + ", ".join(BALANCE_GROUPS) + ". positif / negatif = "
                                                   "images avec / sans ajout ; clip_vide = images des vidéos "
                                                   "sans aucune image avec ajout ; portrait / paysage = "
                                                   "orientation de la vidéo. Ex. {'negatif': 2, 'paysage': 0.7}. "
                                                   "Le dataset n'est pas modifié."},
                        "weights": {"type": "object", "additionalProperties": {"type": "number"},
                                    "description": "Poids des images d'entraînement dans la perte, sans changer "
                                                   "les données : {groupe: poids, > 0 et au plus 10}, mêmes "
                                                   "groupes que balance. Une image dans plusieurs groupes prend "
                                                   "le plus grand poids, les autres gardent 1. S'ajoute à la "
                                                   "pondération équilibrée des classes déjà appliquée (logreg). "
                                                   "Ex. {'clip_vide': 3}."},
                        "note": {"type": "string",
                                 "description": "Hypothèse testée, en une phrase, sans recopier la config."},
                    },
                    "required": ["embeddings"],
                },
                fn=lab.train_model,
            ),
        ],
    )
