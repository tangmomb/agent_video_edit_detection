"""Dataset Builder : génère les vidéos synthétiques annotées et les frames étiquetées."""

from edit_generator.generator import ELEMENT_TYPES
from model_training_pipeline.lab import Lab

from .common import CONTEXT, dataset_info_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Dataset Builder. Tu constitues et enrichis le dataset à la demande de l'orchestrateur. \
Tu n'entraînes ni n'évalues de modèle, et tu ne t'occupes pas des splits train/val/test : c'est le \
Model Researcher qui les constitue à partir du dataset complet. Ton but est un dataset riche et \
varié, où chaque cas (type d'ajout, animation, orientation, taille, fond, thème, vidéos sans ajout) \
est assez représenté pour être appris et mesuré.

Méthode : commence par dataset_info (bloc dataset : composition globale) et l'historique des \
générations, puis génère exactement ce qui est demandé (volume, thèmes, orientation, types \
d'éléments sur-représentés) sans dépasser le budget. Préfère plusieurs petites générations ciblées \
à une grosse générique, et comble en priorité les cas rares. Un clip source n'est jamais \
réutilisé : un thème déjà très exploité (voir diversite.videos_par_theme dans dataset_info) finit \
par ne plus rien donner, et l'erreur « Aucune vidéo Pixabay… » signifie qu'il faut changer de \
thème ou d'orientation.

Exclusions : exclude_videos retire une vidéo de tous les splits. Ne l'utilise que pour un défaut \
de la vidéo elle-même signalé par un autre agent ou constaté dans les données (rendu cassé, \
doublon d'un clip), jamais parce qu'un modèle s'y trompe : les cas où le modèle échoue sont \
précisément ceux qu'il faut garder.

Rapport final (court, en français) : ce que tu as généré et pourquoi, la composition du dataset \
après génération (vidéos, frames, éléments par type et par animation), les cas encore rares, les \
échecs éventuels et le budget restant."""


def make(lab: Lab) -> Agent:
    return Agent(
        name="dataset_builder",
        system=SYSTEM,
        max_turns=12,
        tools=[
            dataset_info_tool(lab),
            Tool(
                name="data_requests",
                description="Historique des générations passées (volume, paramètres, raison).",
                input_schema={"type": "object", "properties": {}},
                fn=lab.data_requests,
            ),
            Tool(
                name="generate_data",
                description="Génère de nouvelles vidéos annotées, chacune sur un clip source jamais "
                            "utilisé, et les ajoute au dataset, puis extrait et étiquette leurs frames. "
                            "Coûteux en temps et en quota d'API d'images : justifie chaque génération.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 1, "maximum": 100},
                        "reason": {"type": "string", "description": "Pourquoi ces données sont nécessaires."},
                        "query": {"type": "string",
                                  "description": "Thème des vidéos sources en anglais (sinon aléatoire)."},
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
            Tool(
                name="exclude_videos",
                description="Retire des vidéos de tous les splits, présents et futurs, sans supprimer de "
                            "fichier. Réservé aux vidéos défectueuses ou dupliquées.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "videos": {"type": "object", "additionalProperties": {"type": "string"},
                                   "description": "{sample_id: raison du défaut}"},
                    },
                    "required": ["videos"],
                },
                fn=lab.exclude_videos,
            ),
        ],
    )
