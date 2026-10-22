"""Contexte et outils partagés par plusieurs agents."""

from selector.lab import Lab

from .runner import Tool

CONTEXT = """Projet : un classifieur d'images qui dit si une frame de vidéo contient un ajout de \
montage (mot-clé, question, sous-titre, bandeau, image incrustée, forme) ou non. Les frames sont \
échantillonnées à 1 par seconde ; une frame est positive si au moins un ajout y est pleinement \
visible. Les frames ambiguës (un ajout en fondu) sont exclues de l'entraînement et de la mesure.

Objectif : le meilleur rappel possible à précision fixée (95 % par défaut) sur le split val, \
c'est-à-dire détecter le plus de frames avec ajout en gardant au plus 5 % de fausses alertes, \
avec un coût de calcul modeste (quelques dizaines de ms/frame sur GPU). Le seuil de décision est \
choisi sur val puis appliqué tel quel sur test. AUC et AP servent de repères secondaires.

Données : vidéos synthétiques (clips libres de droits + overlays aléatoires) dont la vérité \
terrain est connue. Split train/val/test par vidéo source. Les décisions se prennent sur val ; \
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


def dataset_info_tool(lab: Lab) -> Tool:
    return Tool(
        name="dataset_info",
        description="État du dataset : vidéos/frames/éléments par split, taux de frames positives, "
                    "frames ambiguës exclues, extracteurs d'embeddings disponibles et leur coût, "
                    "budgets restants (génération, tests finaux, quotas d'API).",
        input_schema={"type": "object", "properties": {}},
        fn=lab.dataset_info,
    )


def list_experiments_tool(lab: Lab) -> Tool:
    return Tool(
        name="list_experiments",
        description="Historique complet des expériences (toutes sessions) : config (embeddings, modèle, "
                    "hyperparamètres, hypothèse), métriques val (rappel à précision 90/95/99 %, AUC, AP, "
                    "ms/frame), volume de données utilisé (vidéos et frames par split, fps), date, "
                    "résultat du test final et statut de production.",
        input_schema={"type": "object", "properties": {}},
        fn=lab.list_experiments,
    )
