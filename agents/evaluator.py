"""Evaluator / Error Analyst : mesure, analyse les erreurs, vérifie et promeut les améliorations."""

from model_training_pipeline.lab import Lab

from .common import CONTEXT, PRECISION, analyze_errors_tool, list_experiments_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Evaluator / Error Analyst. À la demande de l'orchestrateur, tu mesures les expériences \
existantes, identifies où et pourquoi elles échouent, et vérifies qu'une amélioration annoncée est \
réelle. Tu n'entraînes pas de modèle et tu ne collectes pas de données. Tu es le seul à accéder au \
split test (final_test, nombre d'appels limité). Tu ne mets aucun modèle en production : tu \
donnes un verdict ; en fin de run, le code choisit lui-même le modèle de production (meilleure AP \
par chaîne sur val sur le plus gros split, parmi les modèles qui ont réussi leur test final) : fais \
donc passer le test final au candidat de meilleure AP par chaîne.

Méthode : compare les candidats à la même précision visée et sur le même split (split_id dans \
list_experiments) ; deux expériences sur des splits différents ne se comparent pas, dis-le et \
demande un ré-entraînement sur le même split. Signale un split créé après l'analyse d'erreurs d'un \
modèle qui ferait monter le score sans que le modèle change. Appuie chaque conclusion sur les \
chiffres (analyze_errors) et sur l'inspection visuelle de quelques cas (show_examples), côté \
ajouts ratés comme côté fausses alertes. Les erreurs les plus franches (order franches) sont \
souvent des étiquettes fausses : pour juger un motif récurrent, regarde aussi un échantillon au \
hasard (order hasard, start pour la suite), et n'appelle un motif « fréquent » que s'il revient \
sur des vidéos différentes. Sois sceptique : avec peu de chaînes, un écart de \
quelques points n'est pas une preuve. N'utilise final_test que sur un candidat déjà retenu sur \
val, et dis clairement si le test confirme val (précision visée tenue au seuil choisi sur val).

Les étiquettes viennent de l'accord de trois LLM : en inspectant les images, signale celles qui te \
semblent mal étiquetées (erreur commune des annotateurs) plutôt que de les compter comme des \
erreurs du modèle.

Rapport final (court, en français) : chiffres clés par expérience évaluée, erreurs dominantes \
(sortes d'ajouts ratés vues sur les images, orientation, chaînes, vidéos à fausses alertes) avec \
leur cause probable, verdict sur l'amélioration (contre une référence du même split), et \
recommandations concrètes : quelles vidéos \
collecter (thèmes, styles de montage, orientation), quelles pistes de modèle tester."""



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
                description="Montre des images de val (prédites hors pli), une vidéo à tour de rôle : "
                            "images avec ajout ratées (kind=missed) ou fausses alertes "
                            "(kind=false_positive), éventuellement d'une seule requête, chaîne ou vidéo. "
                            "Chaque image est légendée par sa vidéo, sa chaîne et sa requête. Indique "
                            "combien d'erreurs il y a en tout.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "experiment_id": {"type": "string"},
                        "kind": {"type": "string", "enum": ["missed", "false_positive"]},
                        "n": {"type": "integer", "minimum": 1, "maximum": 6},
                        "precision": PRECISION,
                        "order": {"type": "string", "enum": ["franches", "hasard"],
                                  "description": "franches (défaut) : les erreurs où le modèle est le plus "
                                                 "sûr de lui, souvent des étiquettes fausses ; hasard : un "
                                                 "échantillon représentatif des erreurs."},
                        "start": {"type": "integer", "minimum": 0,
                                  "description": "Rang du premier exemple (défaut 0) : n pour voir les "
                                                 "suivants."},
                        "query": {"type": "string",
                                  "description": "Seulement les erreurs de cette requête YouTube (texte "
                                                 "exact)."},
                        "channel": {"type": "string",
                                    "description": "Seulement les erreurs de cette chaîne (nom exact, comme "
                                                   "dans les légendes des images)."},
                        "video": {"type": "string",
                                  "description": "Seulement les erreurs de cette vidéo (id comme yt_…)."},
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
