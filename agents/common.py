"""Contexte et outils partagés par plusieurs agents."""

from model_training_pipeline.lab import Lab

from .runner import Tool

CONTEXT = """Projet : un classifieur d'images qui dit si une frame de vidéo contient un ajout de \
montage (mot-clé, question, sous-titre, bandeau, image incrustée, forme, carton plein écran \
de type chapitre ou question sur fond opaque) ou non. Les frames sont \
échantillonnées à 1 par seconde ; une frame est positive si au moins un ajout y est pleinement \
visible. Les frames ambiguës (un ajout en fondu) sont exclues de l'entraînement et de la mesure.

Objectif : le meilleur rappel possible à précision fixée (95 % par défaut) sur le split val, \
c'est-à-dire détecter le plus de frames avec ajout en gardant au plus 5 % de fausses alertes, \
avec un coût de calcul modeste (quelques dizaines de ms/frame sur GPU). Le seuil de décision est \
choisi sur val puis appliqué tel quel sur test. AUC et AP servent de repères secondaires.

Données : vidéos synthétiques (rushes bruts libres de droits + overlays aléatoires) dont la vérité \
terrain est connue. Le Dataset Builder les génère sans se soucier des splits ; le Model Researcher \
découpe ensuite le dataset en train/val/test (make_split) : un split est nommé, figé, fait par rush \
brut, et les vidéos générées après sa création n'y entrent pas. Sans split, on ne peut ni \
entraîner ni tester. Chaque expérience retient son split : deux \
expériences ne se comparent que si elles ont le même split. Les décisions se prennent sur val ; \
test est réservé à la vérification finale d'un candidat. Le split val est petit : une différence \
de quelques points peut être du bruit.

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


# Ce que chaque agent lit de dataset_info : le Dataset Builder les thèmes et les budgets de
# génération, le Model Researcher les extracteurs et la composition d'un split.
DATASET_INFO_KEYS = {"dataset_builder": ("dataset", "diversite", "budgets"),
                     "model_researcher": ("fps", "dataset", "extractors", "composition_du_split")}


def dataset_info_tool(lab: Lab, agent: str) -> Tool:
    keys = DATASET_INFO_KEYS[agent]
    props = {}
    if agent == "model_researcher":
        props["split_id"] = {"type": "string",
                             "description": "Ajoute la composition détaillée de train, val et test pour ce "
                                            "split (id renvoyé par make_split)."}

    def fn(split_id: str | None = None) -> dict:
        info = lab.dataset_info(split_id)
        if info.get("dataset_vide"):
            return info
        out = {k: info[k] for k in keys if k in info}
        if "budgets" in out:
            out["budgets"] = {k: v for k, v in out["budgets"].items() if k != "final_tests_restants"}
        return out

    contenu = ("diversité (vidéos par thème, rushes bruts réutilisés), budgets restants (génération, "
               "quotas d'API)" if agent == "dataset_builder" else
               "extracteurs d'embeddings disponibles et leur coût")
    return Tool(
        name="dataset_info",
        description="État du dataset : composition globale (clips générés, rushes bruts uniques, images "
                    "extraites des clips, pourcentage d'images avec ajout, frames ambiguës exclues, clips sans "
                    "et avec ajout, répartition des vidéos par orientation et "
                    "des éléments par type, animation d'entrée, animation de sortie, surface et fond), "
                    + contenu + ".",
        input_schema={"type": "object", "properties": props},
        fn=fn,
    )


def list_experiments_tool(lab: Lab) -> Tool:
    return Tool(
        name="list_experiments",
        description="Splits figés (id, clips générés en train/val/test, note, paramètres, date, clips "
                    "exclus, clips générés après le split) et expériences de toutes les sessions, en "
                    "format court : split, config (embeddings, modèle, hyperparamètres, rééquilibrage de "
                    "train, hypothèse), val (AUC, AP, rappel au seuil de précision 95 % avec FP/FN), "
                    "ms/frame, test final, mise en production.",
        input_schema={"type": "object", "properties": {
            "split_id": {"type": "string", "description": "Ne garder que les expériences de ce split."}}},
        fn=lab.experiments_overview,
    )
