"""Evaluator / Error Analyst : mesure, analyse les erreurs, vérifie et promeut les améliorations."""

from model_training_pipeline.lab import Lab

from .common import CONTEXT, list_experiments_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Evaluator / Error Analyst. À la demande de l'orchestrateur, tu mesures les expériences \
existantes, identifies où et pourquoi elles échouent, et vérifies qu'une amélioration annoncée est \
réelle. Tu n'entraînes pas de modèle et tu ne génères pas de données. Tu es le seul à accéder au \
split test (final_test, nombre d'appels limité) et à mettre un modèle en production.

Méthode : compare les candidats à la même précision visée. Appuie chaque conclusion sur les \
chiffres (analyze_errors) et sur l'inspection visuelle de quelques cas (show_examples), côté \
ajouts ratés comme côté fausses alertes. Sois sceptique : sur un petit split val, un écart de \
quelques points n'est pas une preuve. N'utilise final_test que sur un candidat déjà retenu sur \
val, et save_model seulement si l'orchestrateur te le demande et que le test confirme val.

Rapport final (court, en français) : chiffres clés par expérience évaluée, erreurs dominantes \
(types d'ajout, tailles, fonds, nombre d'ajouts, orientation, vidéos à fausses alertes) avec leur \
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
                            "frames positives ratées par type d'ajout, surface, fond, nombre d'ajouts et "
                            "orientation ; répartition des fausses alertes ; exemples de frames ratées.",
                input_schema={
                    "type": "object",
                    "properties": {"experiment_id": {"type": "string"}, "precision": PRECISION},
                    "required": ["experiment_id"],
                },
                fn=lab.analyze_errors,
            ),
            Tool(
                name="show_examples",
                description="Montre des frames de val, ajouts encadrés en rouge : frames avec ajout ratées "
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
                description="Évalue une expérience sur le split test avec le seuil choisi sur val : rappel, "
                            "précision, TP/FP/FN, AUC, AP. Nombre d'appels limité par session : à réserver au "
                            "candidat final.",
                input_schema={
                    "type": "object",
                    "properties": {"experiment_id": {"type": "string"}, "precision": PRECISION},
                    "required": ["experiment_id"],
                },
                fn=lab.final_test,
            ),
            Tool(
                name="save_model",
                description="Enregistre l'expérience comme modèle de production (modèle + seuil).",
                input_schema={
                    "type": "object",
                    "properties": {"experiment_id": {"type": "string"}, "precision": PRECISION,
                                   "note": {"type": "string"}},
                    "required": ["experiment_id", "precision", "note"],
                },
                fn=lab.save_model,
            ),
        ],
    )
