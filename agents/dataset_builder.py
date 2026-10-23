"""Dataset Builder : génère les clips synthétiques annotés et les images étiquetées."""

from edit_generator.generator import ELEMENT_TYPES
from model_training_pipeline.lab import Lab

from .common import CONTEXT, dataset_info_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Dataset Builder. Tu enrichis le dataset à la demande de l'orchestrateur : tu ne fais \
qu'ajouter des données, tu n'en retires jamais. \
Tu n'entraînes ni n'évalues de modèle, et tu ne t'occupes pas des splits train/val/test : c'est le \
Model Researcher qui les constitue à partir du dataset complet. Ton but est un dataset riche et \
varié, où chaque cas (type d'ajout, animation, orientation, taille, fond, thème, clips sans ajout) \
est assez représenté pour être appris et mesuré.

Méthode : commence par dataset_info (bloc dataset : composition globale) et l'historique des \
générations, puis génère exactement ce qui est demandé (volume, thèmes, orientation, types \
d'éléments sur-représentés) sans dépasser le budget. Préfère plusieurs petites générations ciblées \
à une grosse générique, et comble en priorité les cas rares. Un rush brut n'est jamais \
réutilisé : un thème déjà très exploité (voir diversite.clips_par_theme dans dataset_info) finit \
par ne plus rien donner, et l'erreur « Aucune vidéo Pixabay… » signifie qu'il faut changer de \
thème ou d'orientation. Le thème (query) est une recherche Pixabay : 2 à 4 mots-clés en anglais \
(« neon street night », pas une phrase), car chaque mot en plus réduit les résultats. Il ne sert \
qu'à choisir le rush brut : peu importe que le texte et les photos des ajouts n'aient rien à voir.

Rapport final (court, en français) : ce que tu as généré et pourquoi, la composition du dataset \
après génération (clips, images, éléments par type et par animation), les cas encore rares, les \
échecs éventuels et le budget restant."""


def make(lab: Lab) -> Agent:
    return Agent(
        name="dataset_builder",
        system=SYSTEM,
        max_turns=12,
        tools=[
            dataset_info_tool(lab, "dataset_builder"),
            Tool(
                name="data_requested_history",
                description="Historique des générations passées, regroupé par thème : appels, clips "
                            "demandés, clips obtenus, orientations. Moins de clips obtenus que demandés : "
                            "le thème s'épuise.",
                input_schema={"type": "object", "properties": {}},
                fn=lab.data_requested_history,
            ),
            Tool(
                name="generate_data",
                description="Génère de nouvelles vidéos annotées, chacune sur un rush brut jamais "
                            "utilisé, et les ajoute au dataset, puis extrait et étiquette leurs images. "
                            "Coûteux en temps et en quota d'API d'images : justifie chaque génération.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 1, "maximum": 100},
                        "reason": {"type": "string", "description": "Pourquoi ces données sont nécessaires."},
                        "query": {"type": "string",
                                  "description": "Recherche Pixabay des rushes bruts : 2 à 4 mots-clés "
                                                 "en anglais (sinon thème aléatoire)."},
                        "orientation": {"type": "string", "enum": ["portrait", "landscape"]},
                        "type_weights": {"type": "object", "additionalProperties": {"type": "number"},
                                         "description": "Poids relatifs des types : " + ", ".join(ELEMENT_TYPES)},
                        "lang": {"type": "string", "enum": ["fr", "en", "mix"]},
                        "empty_prob": {"type": "number", "minimum": 0, "maximum": 1,
                                       "description": "Proportion de vidéos sans aucun ajout (négatifs)."},
                    },
                    "required": ["n", "reason"],
                },
                fn=lab.generate_data,
            ),
        ],
    )
