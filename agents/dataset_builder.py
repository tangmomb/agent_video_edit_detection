"""Dataset Builder : génère les vidéos synthétiques annotées et les frames étiquetées."""

from editgen.generator import ELEMENT_TYPES
from selector.lab import Lab

from .common import CONTEXT, dataset_info_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Dataset Builder. Tu constitues et enrichis le dataset à la demande de l'orchestrateur. \
Tu n'entraînes ni n'évalues de modèle.

Méthode : commence par dataset_info et l'historique des générations, puis génère exactement ce qui \
est demandé (volume, thèmes, orientation, types d'éléments sur-représentés) sans dépasser le budget. \
Préfère plusieurs petites générations ciblées à une grosse générique. Vérifie après coup que les \
splits restent équilibrés (assez de vidéos sources en val et test). Sans paramètre split, chaque \
vidéo tombe en train, val ou test selon sa vidéo source (environ 60/20/20) : pour renforcer \
l'entraînement sur un type d'ajout ou des négatifs difficiles, utilise split="train", sinon une \
partie des vidéos ciblées part en val/test. Réserve split="val" ou "test" aux demandes explicites \
d'agrandir ces splits.

Rapport final (court, en français) : ce que tu as généré et pourquoi, l'état du dataset après \
génération (vidéos et frames par split, éléments par type), les échecs éventuels et le budget restant."""


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
                description="Génère de nouvelles vidéos annotées (réparties en train/val/test selon la vidéo "
                            "source, ou toutes dans le split demandé) puis ré-extrait et étiquette les frames. "
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
                        "split": {"type": "string", "enum": ["train", "val", "test"],
                                  "description": "Met toutes les vidéos de cette génération dans ce split, en "
                                                 "ne choisissant que des vidéos sources qui y tombent. Par "
                                                 "défaut : réparties selon la source (~60/20/20)."},
                    },
                    "required": ["n", "reason"],
                },
                fn=lab.generate_data,
            ),
        ],
    )
