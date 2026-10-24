"""Contexte et outils partagés par plusieurs agents."""

from model_training_pipeline.lab import Lab

from .runner import Tool

CONTEXT = """Projet : un classifieur d'images qui dit si une image extraite d'une vidéo contient un ajout de \
montage (habillage, texte, titre, sous-titres, photo ou image incrustée, logo, cadre, effet \
graphique : tout ce qui ne vient pas de la caméra) ou si elle est brute de la caméra. Les images \
sont échantillonnées à 1 par seconde ; une image est positive si au moins un ajout y est visible.

Objectif : le meilleur rappel possible à précision fixée (95 % par défaut) sur le split val, \
c'est-à-dire détecter le plus d'images avec ajout en gardant au plus 5 % de fausses alertes. \
Le temps de calcul n'est pas un critère. Le seuil de décision est choisi sur val puis appliqué tel quel sur test. AUC et AP servent de repères secondaires.

Données : vraies vidéos YouTube. Chaque image est annotée par trois LLM de vision ; elle n'a \
d'étiquette que s'ils sont tous d'accord, sinon elle est écartée (ni entraînée ni mesurée). Les \
étiquettes peuvent donc contenir quelques erreurs communes aux trois LLM, et les cas ambigus \
manquent. Le Dataset Builder collecte les vidéos sans se soucier des splits ; le Model Researcher \
découpe ensuite le dataset en train/val/test (make_split) : un split est nommé, figé, fait par \
chaîne YouTube (une chaîne garde le même habillage d'une vidéo à l'autre), et les vidéos \
collectées après sa création n'y entrent pas. Sans split, on ne peut ni entraîner ni tester. Chaque expérience retient son split : deux \
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


# Ce que chaque agent lit de dataset_info : le Dataset Builder la diversité et les budgets de
# collecte, le Model Researcher la composition d'un split (les extracteurs sont listés dans
# la description de train_model).
DATASET_INFO_KEYS = {"dataset_builder": ("dataset", "diversite", "budgets"),
                     "model_researcher": ("fps", "dataset", "composition_du_split")}


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

    contenu = (", diversité (chaînes les plus représentées), budgets restants (vidéos, "
               "annotation en $)" if agent == "dataset_builder" else
               " ; avec split_id, la même composition pour train, val et test")
    return Tool(
        name="dataset_info",
        description="État du dataset : composition globale (chaînes YouTube, vidéos, images extraites, "
                    "images étiquetées et sans étiquette faute d'accord des annotateurs, pourcentage "
                    "d'images avec ajout, vidéos sans aucun ajout, répartition des vidéos par "
                    "orientation)"
                    + contenu + ".",
        input_schema={"type": "object", "properties": props},
        fn=fn,
    )


def list_experiments_tool(lab: Lab) -> Tool:
    return Tool(
        name="list_experiments",
        description="Splits figés et, sous chacun, ses expériences (toutes sessions). Split : chaînes "
                    "YouTube en train/val/test et chaines_shared (chaînes présentes dans plusieurs "
                    "parties, toujours 0 : une chaîne n'appartient qu'à une partie), vidéos (clips) en "
                    "train/val/test, images étiquetées du val et images positives du val, vidéos "
                    "collectées après le split, "
                    "note. Expérience : config (embeddings, modèle, temporal, hyperparamètres, "
                    "rééchantillonnage, pondération), hypothèse, val (AUC, AP, seuil et rappel à 95 % de précision, FP/FN), "
                    "erreurs_val à ce seuil (clips aux ajouts ratés, clips à fausses alertes), "
                    "test final, en_production si c'est le modèle en production. Un split sans "
                    "expérience n'apparaît que s'il est le plus récent.",
        input_schema={"type": "object", "properties": {
            "split_id": {"type": "string", "description": "Ne renvoyer que ce split et ses expériences."}}},
        fn=lab.experiments_overview,
    )
