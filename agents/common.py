"""Contexte et outils partagés par plusieurs agents."""

from model_training_pipeline import metrics
from model_training_pipeline.lab import Lab

from .runner import Tool

# Contexte de tous les agents ; MEASURE s'y ajoute pour ceux qui entraînent, mesurent ou décident.
CONTEXT = """Projet : un classifieur qui dit si une image de vidéo YouTube contient un ajout de montage \
(texte, sous-titres, logo, bandeau, image incrustée, cadre, effet graphique : tout ce qui ne vient \
pas de la caméra). Une image par seconde de vidéo. Partout, « chaîne » veut dire chaîne YouTube, et \
chaque vidéo vient d'une chaîne différente.

Objectif : le meilleur rappel à 90 % de précision (abrégé « rappel à 90 % »), sur des chaînes \
YouTube que le modèle n'a jamais vues. Le temps de calcul ne compte pas.

Données : chaque image est étiquetée par trois LLM de vision, seulement s'ils sont d'accord.

Règles, sans exception :
- Tu n'agis qu'avec tes outils : ni terminal, ni fichier, ni code. N'écris jamais de commande ou \
de sortie de programme comme si tu l'avais lancée.
- N'affirme que ce que tes outils ont renvoyé : aucun chiffre, fichier, modèle ou donnée inventé.
- Si c'est impossible (budget épuisé, outil en erreur), dis-le en citant l'erreur et arrête-toi. \
Ne retente pas une action refusée en changeant seulement ses paramètres.
- N'appelle pas deux fois de suite le même outil de consultation : son résultat n'a pas changé."""

MEASURE = """

Mesure :
- Un split sépare les vidéos en dev et test, par chaîne. Il ne change plus, et chaque split a son \
propre test.
- val est une validation croisée par chaîne sur dev : chaque image est prédite par un modèle qui \
n'a jamais vu sa chaîne. On décide sur val ; test sert seulement à vérifier un candidat à la fin.
- Le seuil est fixé par le code sur val, prudemment, puis appliqué tel quel sur test.
- AP par chaîne : l'AP (qualité du classement, tous seuils confondus) calculée sur les images de \
val de chaque chaîne, puis moyennée. Elle dit si le modèle reconnaît les ajouts sur des chaînes \
jamais vues.
- Meilleur modèle : parmi ceux dont l'AP par chaîne est à moins de 0,005 de la meilleure, celui \
de meilleur rappel à 90 %. Le code choisit la production ainsi.
- Deux expériences ne se comparent que sur le même split : n'écris jamais d'écart de score entre \
deux splits. Pour mesurer ce qu'apporte un nouveau split, on y réentraîne la meilleure config \
d'avant."""


# Ce que chaque agent lit de dataset_info : le Dataset Builder les budgets de collecte, le Model
# Researcher la composition d'un split (les extracteurs sont listés dans la description de
# train_model).
DATASET_INFO_KEYS = {"dataset_builder": ("dataset_total_disponible", "budgets"),
                     "model_researcher": ("dataset_total_disponible", "composition_du_dernier_split")}
# Comptes d'images qui ne décident de rien pour un agent : toutes les images extraites, et celles
# jamais annotées (sautées comme quasi identiques, voulu) ; restent dans dataset_info pour la CLI.
HIDDEN_COUNTS = ("images_extraites", "images_non_annotees")


def _visible(composition: dict) -> dict:
    return {k: v for k, v in composition.items() if k not in HIDDEN_COUNTS}


def dataset_info_tool(lab: Lab, agent: str) -> Tool:
    keys = DATASET_INFO_KEYS[agent]

    def fn() -> dict:
        # Model Researcher : avec la composition du split le plus récent, celui sur lequel entraîner.
        info = lab.dataset_info(lab.latest_split() if "composition_du_dernier_split" in keys else None)
        if info.get("dataset_vide"):
            return info
        out = {k: info[k] for k in keys if k in info}
        out["dataset_total_disponible"] = _visible(out["dataset_total_disponible"])
        if "composition_du_dernier_split" in out:
            out["composition_du_dernier_split"] = {k: _visible(v) if isinstance(v, dict) else v
                                                   for k, v in out["composition_du_dernier_split"].items()}
        if "budgets" in out:
            out["budgets"] = {k: v for k, v in out["budgets"].items() if k != "final_tests_restants"}
        return out

    contenu = (" et budgets restants (vidéos, annotation)" if agent == "dataset_builder" else
               ", puis celle de dev et test du dernier split et les vidéos hors de ce split")
    return Tool(
        name="dataset_info",
        description="Composition du dataset (vidéos, images étiquetées ou non, part d'images avec "
                    "ajout, orientation)" + contenu + ".",
        input_schema={"type": "object", "properties": {}},
        fn=fn,
    )


def list_experiments_tool(lab: Lab, brief: bool = False) -> Tool:
    """brief : version de l'orchestrateur, qui ne choisit pas les configs (lab.experiments_overview)."""
    shown = ("les 3 meilleures" if brief else
             "les 5 meilleures, celles testées ou en production et les 3 dernières, le reste en un total")
    description = ("Le modèle en production, puis le dernier split : candidat_a_tester (la prochaine "
                   "expérience à passer au test final, choisie par la règle du meilleur modèle) et ses "
                   "expériences : " + shown + ". "
                   "Une ligne par expérience : config → AP par chaîne, rappel à 90 % (FP, FN), test "
                   "final. Pour chaque ancien split, sa meilleure expérience.")
    return Tool(
        name="list_experiments",
        description=description,
        input_schema={"type": "object", "properties": {}},
        fn=lambda: lab.experiments_overview(brief),
    )


PRECISION = {"type": "number", "description": "Précision visée, qui fixe le seuil (défaut 0.9)."}


# Analyse visuelle d'analyze_errors, à chaque appel : chaque erreur val est décrite par un LLM de
# vision (lab.describe_errors, en batch, mis en cache par image), puis le modèle de l'agent résume
# ces descriptions en points communs. L'agent qui appelle l'outil ne reçoit que le résumé.
SUMMARY_PROMPT = """Tu analyses les erreurs d'un classifieur d'images qui dit si une image extraite \
d'une vidéo YouTube contient un ajout de montage (habillage, texte, titre, sous-titres, logo, \
bandeau, image incrustée, cadre, effet graphique : tout ce qui ne vient pas de la caméra). Voici \
toutes ses erreurs sur val, une par ligne : ajouts ratés et fausses alertes (images brutes prises \
pour des images avec ajout), avec la description de l'image par un LLM de vision.

Réponds en français, en trois lignes, sans nommer de vidéo ni de chaîne :
Ajouts ratés : les 1 à 3 types d'ajout qui reviennent très souvent (au moins un ajout raté sur \
dix), chacun en quelques mots avec sa position et son nombre sur {fn}, du plus fréquent au moins \
fréquent (exemple : « logo petit en coin haut droit (40/{fn}) ; sous-titres blancs en bas \
(25/{fn}) »).
Fausses alertes : de même, ce qui revient très souvent dans ces images brutes et ressemble à un \
ajout, avec son nombre sur {fp}.
Étiquettes douteuses : le nombre d'erreurs dont la description contredit l'étiquette.
Ignore les cas isolés ; « aucun type fréquent » si rien ne revient souvent. Appuie-toi seulement \
sur ces descriptions.

Erreurs :
"""


def analyze_errors_tool(lab: Lab, session=None) -> Tool:
    """session : celle du run, pour l'analyse visuelle (sans session, l'outil s'en passe)."""
    def fn(experiment_id: str, precision: float = metrics.DEFAULT_PRECISION) -> dict:
        out = lab.analyze_errors(experiment_id, precision)
        if session is None:
            return out
        head = {"experiment_id": out.pop("experiment_id")}
        left = session.budget_usd - session.reserve_usd - session.spent
        if left <= 0:
            return {**head, "analyse_visuelle": "non faite : budget de session épuisé", **out}
        try:
            d = lab.describe_errors(experiment_id, precision, max_usd=left)
        except ValueError as e:
            return {**head, "analyse_visuelle": f"non faite : {e}", **out}
        session.add_cost(d["cost_usd"])
        # sans vidéo ni chaîne : le résumé doit dire quels ajouts reviennent, pas où
        lines = "\n".join(f"{e['erreur']} | {e['description'] or '(non décrite)'}" for e in d["erreurs"])
        prompt = SUMMARY_PROMPT.format(fn=out["fn"], fp=out["fp"])
        summary, cost = session.look([{"type": "text", "text": prompt + lines}])
        # en tête de la réponse : c'est ce qui dit quoi corriger, les chiffres suivent
        return {**head, "analyse_visuelle": summary,
                "cout_analyse_visuelle_usd": round(d["cost_usd"] + cost, 4), **out}

    return Tool(
        name="analyze_errors",
        description=("Erreurs val d'une expérience" + (
                        " : résumé des ajouts ratés et des fausses alertes fréquents, d'après une "
                        "description de chaque image erreur par un LLM de vision, puis" if session else " :")
                     + " rappel, précision, TP/FP/FN et erreurs par orientation."
                     + (" Payant (quelques centimes) ; la première fois, quelques minutes." if session else "")),
        input_schema={"type": "object", "properties": {"experiment_id": {"type": "string"}, "precision": PRECISION},
                      "required": ["experiment_id"]},
        fn=fn,
    )
