"""Evaluator / Error Analyst : mesure, analyse les erreurs, vérifie et promeut les améliorations."""

from model_training_pipeline.lab import Lab

from .common import CONTEXT, MEASURE, PRECISION, analyze_errors_tool, list_experiments_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + MEASURE + """

Ton rôle : Evaluator. Tu analyses les expériences existantes, trouves pourquoi elles échouent et \
vérifies qu'une amélioration annoncée est réelle. Tu n'entraînes pas et ne collectes pas. Tu es le \
seul à accéder au test (final_test, nombre limité).

Méthode :
- Compare les candidats sur le même split ; sinon dis-le et demande un réentraînement.
- Appuie chaque conclusion sur analyze_errors et sur quelques images (show_examples), ajouts ratés \
comme fausses alertes.
- Les erreurs les plus franches sont souvent des étiquettes fausses : regarde aussi un échantillon \
au hasard (order hasard), et n'appelle un motif fréquent que s'il revient sur plusieurs vidéos. \
Signale les images mal étiquetées au lieu de les compter comme erreurs du modèle.
- Signale un split créé après l'analyse d'erreurs d'un modèle : il ferait monter le score sans que \
le modèle change.
- final_test : seulement sur candidat_a_tester (list_experiments), un à la fois. S'il échoue, \
relis list_experiments : le candidat suivant y apparaît. Arrête-toi au premier test réussi. Dis \
clairement si le test confirme val (précision visée tenue). Tu ne mets rien en production : le \
code choisit parmi les modèles qui ont réussi leur test.

Rapport final, court, en français : chiffres clés par expérience, erreurs dominantes et leur cause \
probable, verdict sur l'amélioration (contre une référence du même split), recommandations \
concrètes (vidéos à collecter, pistes de modèle)."""



def make(lab: Lab, session=None) -> Agent:
    return Agent(
        name="evaluator",
        system=SYSTEM,
        max_turns=15,
        tools=[
            list_experiments_tool(lab),
            analyze_errors_tool(lab, session),
            Tool(
                name="show_examples",
                description="Montre des images erreurs de val, légendées (vidéo, chaîne, requête), et leur "
                            "nombre total.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "experiment_id": {"type": "string"},
                        "kind": {"type": "string", "enum": ["missed", "false_positive"],
                                 "description": "missed : ajouts ratés ; false_positive : fausses alertes."},
                        "n": {"type": "integer", "minimum": 1, "maximum": 6},
                        "precision": PRECISION,
                        "order": {"type": "string", "enum": ["franches", "hasard"],
                                  "description": "franches (défaut) : celles où le modèle est le plus sûr "
                                                 "de lui ; hasard : un échantillon représentatif."},
                        "start": {"type": "integer", "minimum": 0, "description": "Rang du premier (défaut 0)."},
                        "query": {"type": "string", "description": "Filtre : requête YouTube exacte."},
                        "channel": {"type": "string", "description": "Filtre : nom exact de la chaîne."},
                        "video": {"type": "string", "description": "Filtre : id de la vidéo (yt_…)."},
                    },
                    "required": ["experiment_id"],
                },
                fn=lab.show_examples,
            ),
            Tool(
                name="final_test",
                description="Mesure une expérience sur le test de son split, au seuil choisi sur val : "
                            "rappel, précision, TP/FP/FN. Nombre limité par session.",
                input_schema={
                    "type": "object",
                    "properties": {"experiment_id": {"type": "string"}, "precision": PRECISION},
                    "required": ["experiment_id"],
                },
                fn=lab.final_test,
            ),
        ],
    )
