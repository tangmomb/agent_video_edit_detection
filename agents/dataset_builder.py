"""Dataset Builder : collecte de vraies vidéos YouTube, annotées par trois LLM, et leurs images étiquetées."""

from model_training_pipeline.lab import Lab

from .common import CONTEXT, dataset_info_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Dataset Builder. Tu ajoutes des vidéos au dataset à la demande de l'orchestrateur ; tu \
n'en retires jamais, et tu ne t'occupes ni des modèles ni des splits.

But : un dataset varié. Beaucoup de chaînes, des styles de montage différents (très habillé, \
sous-titré, sobre), du portrait (Shorts) et du paysage, des images brutes autant qu'avec ajout.

Méthode :
- Commence par dataset_info et data_requested_history.
- Collecte le volume demandé sans dépasser les budgets, en plusieurs collectes de requêtes très \
différentes (thèmes, langues, pays, formats), en comblant d'abord ce que dataset_info montre de \
rare : orientation, vidéos sans ajout.
- La requête est une recherche YouTube de quelques mots (« recette pâtes rapide ») : elle fixe le \
sujet, pas le montage, et rien ne vérifie ce qu'elle rapporte. Ne cherche donc pas à viser un type \
de montage : vise des sujets et des chaînes que le dataset n'a pas encore.
- Une collecte ne prend qu'une vidéo par chaîne nouvelle : une requête très exploitée finit par ne \
plus rien donner. « aucune nouvelle vidéo YouTube » : change de requête ou d'orientation.
- L'annotation est payante et peut prendre de quelques minutes à des heures : le budget \
d'annotation limite le volume.

Rapport final, court, en français : ce que tu as collecté et pourquoi, le dataset après collecte \
(chaînes, vidéos, images étiquetées et sans étiquette, orientation), ce qui reste rare, les échecs \
et les budgets restants."""


def make(lab: Lab) -> Agent:
    return Agent(
        name="dataset_builder",
        system=SYSTEM,
        max_turns=12,
        tools=[
            dataset_info_tool(lab, "dataset_builder"),
            Tool(
                name="data_requested_history",
                description="Requêtes YouTube qui ont amené des vidéos, avec leur nombre par orientation, "
                            "des plus productives aux moins productives.",
                input_schema={"type": "object", "properties": {}},
                fn=lab.data_requested_history,
            ),
            Tool(
                name="generate_data",
                description="Collecte n vidéos YouTube nouvelles (5 min au plus) pour une recherche et fait "
                            "annoter leurs images ; attend la fin de l'annotation. Renvoie les vidéos et images "
                            "obtenues et les budgets restants.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 1, "maximum": 20},
                        "reason": {"type": "string", "description": "Pourquoi ces données."},
                        "query": {"type": "string",
                                  "description": "Quelques mots-clés. En portrait, ils deviennent un hashtag : "
                                                 "préfère un ou deux mots courants (« recette »)."},
                        "orientation": {"type": "string", "enum": ["portrait", "landscape"],
                                        "description": "portrait : Shorts ; sinon recherche classique (surtout "
                                                       "du paysage)."},
                    },
                    "required": ["n", "reason", "query"],
                },
                fn=lab.generate_data,
            ),
        ],
    )
