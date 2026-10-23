"""Evaluator / Error Analyst : mesure, analyse les erreurs, vérifie et promeut les améliorations."""

from model_training_pipeline.lab import Lab

from .common import CONTEXT, list_experiments_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Evaluator / Error Analyst. À la demande de l'orchestrateur, tu mesures les expériences \
existantes, identifies où et pourquoi elles échouent, et vérifies qu'une amélioration annoncée est \
réelle. Tu n'entraînes pas de modèle et tu ne génères pas de données. Tu es le seul à accéder au \
split test (final_test, nombre d'appels limité). Tu ne mets aucun modèle en production : tu \
donnes un verdict ; en fin de run, le code choisit lui-même le modèle de production (meilleur rappel \
val à précision 95 % sur le plus gros split).

Méthode : compare les candidats à la même précision visée et sur le même split (split_id dans \
list_experiments) ; deux expériences sur des splits différents ne se comparent pas, dis-le et \
demande un ré-entraînement sur le même split. Signale un split créé après l'analyse d'erreurs d'un \
modèle qui ferait monter le score sans que le modèle change. Appuie chaque conclusion sur les \
chiffres (analyze_errors) et sur l'inspection visuelle de quelques cas (show_examples), côté \
ajouts ratés comme côté fausses alertes. Sois sceptique : sur un petit split val, un écart de \
quelques points n'est pas une preuve. N'utilise final_test que sur un candidat déjà retenu sur \
val, et dis clairement si le test confirme val (précision visée tenue au seuil choisi sur val).

Rapport final (court, en français) : chiffres clés par expérience évaluée, erreurs dominantes \
(types d'ajout, tailles, fonds, animations et phase d'animation, nombre d'ajouts, orientation, \
vidéos à fausses alertes) avec leur \
cause probable, verdict sur l'amélioration, et recommandations concrètes : quelles données \
générer, quelles pistes de modèle tester."""

PRECISION = {"type": "number", "description": "Précision visée sur val, qui fixe le seuil (défaut 0.95)."}


def make(lab: Lab) -> Agent:
    return Agent(
        name="evaluator",
        system=SYSTEM,
        max_turns=15,
        tools=[
            list_experiments_tool(lab),
            Tool(
                name="analyze_errors",
                description="Au seuil qui atteint la précision visée sur val : rappel, TP/FP/FN, taux de "
                            "images positives ratées par type d'ajout, surface, fond, animation d'entrée, "
                            "phase (entrée ou sortie en cours, repos), nombre d'ajouts et orientation ; "
                            "répartition des fausses alertes ; exemples d'images ratées.",
                input_schema={
                    "type": "object",
                    "properties": {"experiment_id": {"type": "string"}, "precision": PRECISION},
                    "required": ["experiment_id"],
                },
                fn=lab.analyze_errors,
            ),
            Tool(
                name="show_examples",
                description="Montre des images de val, ajouts encadrés en rouge : images avec ajout ratées "
                            "(kind=missed) ou fausses alertes (kind=false_positive).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "experiment_id": {"type": "string"},
                        "kind": {"type": "string", "enum": ["missed", "false_positive"]},
                        "n": {"type": "integer", "minimum": 1, "maximum": 6},
                        "precision": PRECISION,
                    },
                    "required": ["experiment_id"],
                },
                fn=lab.show_examples,
            ),
            Tool(
                name="final_test",
                description="Évalue une expérience sur le test de son split avec le seuil choisi sur val : rappel, "
                            "précision, TP/FP/FN, AUC, AP. Nombre d'appels limité par session : à réserver au "
                            "candidat final.",
                input_schema={
                    "type": "object",
                    "properties": {"experiment_id": {"type": "string"}, "precision": PRECISION},
                    "required": ["experiment_id"],
                },
                fn=lab.final_test,
            ),
        ],
    )
