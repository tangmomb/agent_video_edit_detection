"""Model Researcher : choisit les représentations et les modèles, entraîne, mène les expériences."""

from model_training_pipeline.features import EXTRACTORS
from model_training_pipeline.lab import MODELS, Lab

from .common import CONTEXT, dataset_info_tool, list_experiments_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Model Researcher. Tu mènes les expériences demandées par l'orchestrateur : choix des \
embeddings figés (CLIP seul, DINOv2 seul ou les deux concaténés ; image entière ou quadrants), du \
classifieur et de ses hyperparamètres, de l'information temporelle. Tu ne génères pas de données et \
tu n'accèdes pas au split test.

Méthode : consulte l'historique pour ne pas refaire une expérience existante. Formule une hypothèse \
par expérience (champ note) et change un facteur à la fois pour que les résultats soient comparables. \
Arrête-toi quand le lot demandé est fait ou quand les résultats ne bougent plus.

Rapport final (court, en français) : pour chaque expérience, son id, l'hypothèse et le résultat \
(rappel à 95 % de précision, AUC, AP, ms/frame) ; le meilleur candidat selon toi et pourquoi ; ce \
que tu essaierais ensuite et les données qui te manquent, le cas échéant."""


def make(lab: Lab) -> Agent:
    return Agent(
        name="model_researcher",
        system=SYSTEM,
        max_turns=20,
        tools=[
            dataset_info_tool(lab),
            list_experiments_tool(lab),
            Tool(
                name="train_model",
                description="Entraîne un classifieur binaire (ajout de montage visible ou non) sur des "
                            "embeddings figés et l'évalue sur val. Renvoie l'id d'expérience, le rappel "
                            "(avec seuil et TP/FP/FN) aux précisions 90/95/99 %, AUC, AP et le coût de "
                            "calcul (ms/frame). Les embeddings absents du cache sont calculés à la volée (GPU).",
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
                                     "description": "diff ajoute e_t - e_(t-1) (changement depuis la frame "
                                                    "précédente)."},
                        "params": {"type": "object",
                                   "description": "logreg: C ; mlp: hidden (liste), alpha ; "
                                                  "hgb: max_iter, lr, max_leaf_nodes ; tous: pca (int)."},
                        "note": {"type": "string", "description": "Hypothèse testée."},
                    },
                    "required": ["features"],
                },
                fn=lab.train_model,
            ),
        ],
    )
