"""Dataset Builder : collecte de vraies vidéos YouTube, annotées par trois LLM, et leurs images étiquetées."""

from model_training_pipeline.lab import Lab

from .common import CONTEXT, dataset_info_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + """

Ton rôle : Dataset Builder. Tu enrichis le dataset à la demande de l'orchestrateur : tu ne fais \
qu'ajouter des données, tu n'en retires jamais. \
Tu n'entraînes ni n'évalues de modèle, et tu ne t'occupes pas des splits dev/test : c'est le \
Model Researcher qui les constitue à partir du dataset complet. Ton but est un dataset riche et \
varié : beaucoup de chaînes différentes, des styles de montage variés (vidéos très habillées, \
sous-titrées, sobres), des vidéos portrait (Shorts) et paysage, et des images brutes de caméra \
autant que des images avec ajout.

Méthode : commence par dataset_info (composition globale, chaînes ayant plusieurs vidéos, budgets) \
et l'historique des collectes, puis collecte exactement ce qui est demandé (volume, thèmes, \
orientation) sans dépasser les budgets. Préfère plusieurs petites collectes ciblées à une grosse \
générique, et comble en priorité les cas rares. Une vidéo n'est jamais reprise, et une collecte ne \
garde qu'une vidéo par chaîne absente du dataset : une requête déjà très exploitée (voir data_requested_history) finit par ne plus rien donner, et l'erreur « aucune \
nouvelle vidéo YouTube » signifie qu'il faut changer de requête ou d'orientation. La requête \
(query) est une recherche YouTube : quelques mots-clés (« recette pâtes rapide », « vlog voyage \
japon »), qui fixent le thème et donc, indirectement, le style de montage. Chaque image collectée \
est annotée par trois LLM (payant ; en batch selon la session, l'attente peut durer de quelques \
minutes à plusieurs heures), sauf les images presque identiques à la précédente annotée : le \
budget d'annotation limite le volume, et chaque collecte attend les résultats jusqu'au bout.

Rapport final (court, en français) : ce que tu as collecté et pourquoi, la composition du dataset \
après collecte (chaînes, vidéos, images étiquetées, part d'images sans accord des annotateurs, \
orientation), les cas encore rares, les échecs éventuels et les budgets restants."""


def make(lab: Lab) -> Agent:
    return Agent(
        name="dataset_builder",
        system=SYSTEM,
        max_turns=12,
        tools=[
            dataset_info_tool(lab, "dataset_builder"),
            Tool(
                name="data_requested_history",
                description="Historique des collectes passées, regroupé par requête YouTube : "
                            "appels, vidéos demandées, vidéos obtenues, orientations, coût d'annotation. "
                            "Moins de vidéos obtenues que demandées : la requête s'épuise.",
                input_schema={"type": "object", "properties": {}},
                fn=lab.data_requested_history,
            ),
            Tool(
                name="generate_data",
                description="Collecte de nouvelles vraies vidéos YouTube (jamais prises, 5 min au plus, une "
                            "seule par chaîne absente du dataset) "
                            "pour une recherche, les ajoute au dataset, extrait leurs images (1 par seconde) "
                            "et les fait annoter par trois LLM, en batch ou en appels directs selon la "
                            "session (une image n'est étiquetée que s'ils sont d'accord ; les images presque "
                            "identiques à la précédente annotée sont sautées), puis attend les résultats dans "
                            "le même appel, aussi longtemps qu'il faut. Coûteux en temps et en budget d'annotation : "
                            "justifie chaque collecte.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 1, "maximum": 20},
                        "reason": {"type": "string", "description": "Pourquoi ces données sont nécessaires."},
                        "query": {"type": "string",
                                  "description": "Recherche YouTube : quelques mots-clés (thème, type de "
                                                 "vidéo). En portrait, elle devient un hashtag (mots "
                                                 "collés, sans accents) : préfère alors un ou deux mots "
                                                 "courants (« recette », « astuce cuisine »)."},
                        "orientation": {"type": "string", "enum": ["portrait", "landscape"],
                                        "description": "portrait : Shorts du hashtag ; landscape ou sans "
                                                       "valeur : recherche classique (surtout du paysage)."},
                    },
                    "required": ["n", "reason", "query"],
                },
                fn=lab.generate_data,
            ),
        ],
    )
