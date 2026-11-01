"""Contexte et outils partagés par plusieurs agents."""

from model_training_pipeline import metrics
from model_training_pipeline.lab import Lab

from .runner import Tool

CONTEXT = """Projet : un classifieur d'images qui dit si une image extraite d'une vidéo contient un ajout de \
montage (habillage, texte, titre, sous-titres, photo ou image incrustée, logo, cadre, effet \
graphique : tout ce qui ne vient pas de la caméra) ou si elle est brute de la caméra. Les images \
sont échantillonnées à 1 par seconde ; une image est positive si au moins un ajout y est visible.

Objectif : le meilleur rappel possible à précision fixée (95 % par défaut) sur val, \
c'est-à-dire détecter le plus d'images avec ajout en gardant au plus 5 % de fausses alertes. \
Le temps de calcul n'est pas un critère. Le seuil de décision est choisi sur val puis appliqué tel quel sur test. \
Pour comparer des modèles, l'indicateur de référence est l'AP par chaîne sur val (moyenne des AP de \
chaque chaîne : le modèle classe-t-il bien les images d'une chaîne qu'il n'a jamais vue ?) : c'est sur \
elle que le code choisit le modèle de production. Le rappel val à 95 %, calculé sur toutes les chaînes \
mélangées, favorise des modèles qui ne tiennent pas sur des chaînes nouvelles. AUC et AP globales \
servent de repères secondaires.

Données : vraies vidéos YouTube. Chaque image est annotée par trois LLM de vision ; elle n'a \
d'étiquette que s'ils sont tous d'accord, sinon elle est écartée (ni entraînée ni mesurée). Les \
étiquettes peuvent donc contenir quelques erreurs communes aux trois LLM, et les cas ambigus \
manquent. Le Dataset Builder collecte les vidéos sans se soucier des splits ; le Model Researcher \
découpe ensuite le dataset en dev/test (make_split) : un split est nommé, fait par \
chaîne YouTube (une chaîne garde le même habillage d'une vidéo à l'autre) et ne change plus ; les \
vidéos collectées après sa création n'y entrent pas. Chaque split tire son propre test. Sans split, \
on ne peut ni entraîner ni tester. Chaque expérience retient son split : deux expériences ne se \
comparent que si elles ont le même split, en val comme en test. Pour savoir si un nouveau split \
apporte un gain, on ré-entraîne la meilleure config de l'ancien split sur le nouveau et on compare \
sur celui-ci : n'écris jamais d'écart de score entre deux splits. val est une validation croisée par \
chaîne sur dev : chaque image de dev est prédite par un modèle qui n'a vu aucune vidéo de sa \
chaîne, ce qui mesure le comportement sur des chaînes nouvelles. Le seuil y est choisi prudemment \
(il doit tenir la précision visée sur la plupart des tirages de chaînes) ; le modèle gardé est \
la moyenne des modèles de la validation croisée. Les décisions se prennent sur val ; test est réservé à la \
vérification finale d'un candidat. Le dataset compte peu de chaînes : une différence de quelques \
points peut être du bruit.

Règles d'action, sans exception :
- Tu n'agis sur le projet qu'à travers tes outils. Tu n'as ni terminal, ni accès aux fichiers, ni \
moyen d'écrire ou d'exécuter du code : n'écris jamais de commande, de script ou de sortie de \
programme comme si tu l'avais lancé.
- Ton rapport ne contient que ce que tes outils ont réellement renvoyé. N'invente aucun chiffre, \
aucun fichier, aucun modèle, aucune donnée. Ce que tu n'as pas obtenu par un outil, tu ne l'affirmes pas.
- Si la tâche est impossible avec tes outils (budget épuisé, données absentes, outil en erreur), \
dis-le en une ou deux phrases en citant l'erreur, puis arrête-toi. Ne retente pas une action \
refusée en changeant seulement ses paramètres.
- N'appelle pas deux fois de suite le même outil de consultation : son résultat ne change pas tant \
que rien d'autre n'a été fait."""


# Ce que chaque agent lit de dataset_info : le Dataset Builder les budgets de collecte, le Model
# Researcher la composition d'un split (les extracteurs sont listés dans la description de
# train_model).
DATASET_INFO_KEYS = {"dataset_builder": ("dataset", "budgets"),
                     "model_researcher": ("dataset", "composition_du_split")}
# Comptes d'images qui ne décident de rien pour un agent : toutes les images extraites, et celles
# jamais annotées (sautées comme quasi identiques, voulu) ; restent dans dataset_info pour la CLI.
HIDDEN_COUNTS = ("images_extraites", "images_non_annotees")


def _visible(composition: dict) -> dict:
    return {k: v for k, v in composition.items() if k not in HIDDEN_COUNTS}


def dataset_info_tool(lab: Lab, agent: str) -> Tool:
    keys = DATASET_INFO_KEYS[agent]

    def fn() -> dict:
        # Model Researcher : avec la composition du split le plus récent, celui sur lequel entraîner.
        info = lab.dataset_info(lab.latest_split() if "composition_du_split" in keys else None)
        if info.get("dataset_vide"):
            return info
        out = {k: info[k] for k in keys if k in info}
        out["dataset"] = _visible(out["dataset"])
        if "composition_du_split" in out:
            out["composition_du_split"] = {k: _visible(v) if isinstance(v, dict) else v
                                           for k, v in out["composition_du_split"].items()}
        if "budgets" in out:
            out["budgets"] = {k: v for k, v in out["budgets"].items() if k != "final_tests_restants"}
        return out

    contenu = (", budgets restants (vidéos, "
               "annotation en $ et nombre estimé de vidéos qu'elle permet d'annoter)" if agent == "dataset_builder" else
               " ; la même composition pour dev et test du split le plus récent, s'il y en a un")
    return Tool(
        name="dataset_info",
        description="État du dataset : composition globale (chaînes YouTube, vidéos, images "
                    "étiquetées et sans étiquette faute d'accord des annotateurs, pourcentage "
                    "d'images avec ajout, vidéos sans aucun ajout, répartition des vidéos par "
                    "orientation, chaînes ayant plusieurs vidéos : 15 au plus et un total pour le reste)"
                    + contenu + ".",
        input_schema={"type": "object", "properties": {}},
        fn=fn,
    )


# Construction du split le plus récent : sert à qui fait les splits (Model Researcher) et à qui
# mesure (Evaluator), pas à l'orchestrateur, qui n'en retient que clips_hors_split (faut-il un
# nouveau split ?), la note et les expériences.
SPLIT_DETAIL = ("chaines_dev", "chaines_test", "clips_dev", "clips_test", "images_dev",
                "images_dev_positives", "clips_exclus", "anomalie_chaines_dans_dev_et_test")


def list_experiments_tool(lab: Lab, brief: bool = False) -> Tool:
    def fn() -> dict:
        out = lab.experiments_overview()
        if brief:
            out["splits"] = [{k: v for k, v in sp.items() if k not in SPLIT_DETAIL} for sp in out["splits"]]
        return out

    split = ("vidéos collectées après le split, note" if brief else
             "chaînes YouTube et vidéos (clips) en dev/test, images étiquetées et positives de dev, "
             "vidéos collectées après le split, note")
    return Tool(
        name="list_experiments",
        description="Le split le plus récent (celui sur lequel entraîner) et ses expériences (toutes "
                    "sessions : les 8 meilleures en AP par chaîne, la production, celles passées au test "
                    "final et les 5 dernières en détail, les autres en une ligne dans autres_experiences : "
                    "config et résultat val), puis une ligne par ancien split : nombre d'expériences, la meilleure "
                    "(AP par chaîne en val) et le modèle en production s'il en vient, avec leur config pour "
                    "les refaire sur le split récent et le statut de leur test final (la production "
                    "passe d'abord par un test réussi). Split : " + split + ". Expérience : config (embeddings, modèle, "
                    "hyperparamètres, rééchantillonnage, pondération), hypothèse, val (AP, AP par "
                    "chaîne, rappel à 95 % de précision, FP/FN), test final, en_production si c'est "
                    "le modèle en production.",
        input_schema={"type": "object", "properties": {}},
        fn=fn,
    )


PRECISION = {"type": "number", "description": "Précision visée sur val, qui fixe le seuil (défaut 0.95)."}


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
        description="Au seuil qui atteint la précision visée sur val : rappel, précision, TP/FP/FN et "
                    "erreurs par orientation (paysage, portrait)"
                    + (", précédés d'une analyse visuelle des erreurs typiques : chaque erreur val est "
                       "décrite par un LLM de vision (description gardée pour les appels suivants), puis "
                       "ces descriptions sont résumées (types et positions des ajouts ratés, ce qui "
                       "déclenche les fausses alertes, étiquettes douteuses). Payant à chaque appel "
                       "(quelques centimes sur le budget de session ; la première fois sur une "
                       "expérience, peut attendre quelques minutes)." if session else "."),
        input_schema={"type": "object", "properties": {"experiment_id": {"type": "string"}, "precision": PRECISION},
                      "required": ["experiment_id"]},
        fn=fn,
    )
