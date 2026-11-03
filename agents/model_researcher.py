"""Model Researcher : choisit les représentations et les modèles, entraîne, mène les expériences."""

from model_training_pipeline.features import AGENT_VIEWS, EXTRACTORS
from model_training_pipeline.lab import BALANCE_GROUPS, MODELS, STRATIFY_KEYS, Lab

from .common import CONTEXT, MEASURE, analyze_errors_tool, dataset_info_tool, list_experiments_tool
from .runner import Agent, Tool

SYSTEM = CONTEXT + MEASURE + """

Ton rôle : Model Researcher. Ton but : le meilleur rappel à 90 % de précision, avec un modèle \
entraîné sur le plus de données possible. Tu crées les splits et tu mènes les expériences \
demandées par l'orchestrateur. Tu ne collectes pas de données et tu n'accèdes pas au test.

Splits :
- Crée un split s'il n'y en a pas, ou dès que dataset_info montre des vidéos hors du dernier \
(videos_hors_split > 0) : sinon elles ne servent pas. Tu entraînes toujours sur le dernier split.
- Vérifie sa composition (dataset_info) avant d'entraîner. Garde la stratification par défaut sauf \
raison précise.
- Ne refais jamais un split pour améliorer un score : un découpage choisi pour être facile ne \
mesure plus rien.

Expériences :
- Tu règles les embeddings (CLIP, DINOv2 ou les deux ; image entière, quadrants, bandes ou une \
combinaison, sans répéter un embedding), le classifieur et ses hyperparamètres, le rééchantillonnage (balance) et la pondération (weights) \
d'un groupe difficile.
- Deux classifieurs, logreg et mlp, à égalité : aucun n'est le défaut. Seul val les départage, \
sur le split en cours ; un résultat d'un ancien split ne vaut plus pour celui-ci. Change de \
classifieur comme d'un autre facteur, en gardant le reste de la config.
- Pondérer tous les positifs ou tous les négatifs ne fait surtout que déplacer le seuil : vise un \
groupe précis.
- Regarde list_experiments pour ne pas refaire une expérience. Une hypothèse d'une phrase par \
expérience (champ note, sans recopier la config), un seul facteur changé à la fois.
- Après un nouveau split, réentraîne d'abord la meilleure config précédente avec chacun des deux \
classifieurs (frein réglé pour chacun) : ce sont tes deux références, et la meilleure sur val est \
celle à améliorer.
- Tu ne vois pas les images : analyze_errors te dit quels ajouts tu rates et ce qui déclenche les \
fausses alertes. Appelle-le sur ta référence, puis sur les expériences qui visent un type \
d'erreur : un changement ne vaut que si ces erreurs baissent sans que d'autres montent. Il est \
payant : pas à chaque expérience.
- Arrête-toi quand le lot demandé est fait ou que les résultats ne bougent plus.

Rapport final, court, en français : le split (et pourquoi si tu l'as créé) ; par expérience, id, \
hypothèse, AP par chaîne, rappel à 90 % et effet sur l'erreur visée ; ton meilleur candidat \
(règle de la partie Mesure) et pourquoi ; ce que tu essaierais ensuite et les données qui \
manquent."""


def make(lab: Lab, session=None) -> Agent:
    return Agent(
        name="model_researcher",
        system=SYSTEM,
        max_turns=20,
        tools=[
            # dans l'ordre où il s'en sert le plus souvent
            dataset_info_tool(lab, "model_researcher"),
            list_experiments_tool(lab),
            Tool(
                name="make_split",
                description="Crée un split sur toutes les vidéos actuelles : un test (150 chaînes au plus), "
                            "le reste en dev, découpé en plis par chaîne. Renvoie son id et sa composition.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "note": {"type": "string", "description": "Pourquoi ce split."},
                        "test_ratio": {"type": "number", "minimum": 0.05, "maximum": 0.4,
                                       "description": "Part des chaînes en test (défaut 0.2)."},
                        "stratify": {"type": "array", "items": {"type": "string", "enum": list(STRATIFY_KEYS)},
                                     "description": "Caractéristiques réparties à parts égales entre dev, test "
                                                    "et plis (clip_vide : vidéo sans aucun ajout). Défaut : "
                                                    "toutes ; [] : tirage au hasard."},
                        "seed": {"type": "integer", "description": "Graine du tirage (défaut 0)."},
                    },
                    "required": ["note"],
                },
                fn=lab.make_split,
            ),
            Tool(
                name="train_model",
                description="Entraîne un classifieur sur des embeddings figés et le mesure sur val du dernier "
                            "split. Renvoie l'id d'expérience, l'AP par chaîne, le rappel à 90 % (FP, FN) et "
                            "l'effet de balance et weights.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "embeddings": {"type": "array", "minItems": 1, "uniqueItems": True,
                                     "items": {"type": "string",
                                               "enum": [f"{e}:{v}" for e in EXTRACTORS for v in AGENT_VIEWS]},
                                     "description": "'extracteur:vue', autant que tu veux, chacun une fois, "
                                                    "concaténés dans l'ordre donné. Vues : full (image "
                                                    "entière), quads (ses 4 quadrants seuls, chacun avec un "
                                                    "coin : logos), tb (ses tiers bas et haut seuls : "
                                                    "sous-titres, bandeaux). quads et tb n'ont pas l'image "
                                                    "entière : ajoute full pour l'avoir. Anciennes "
                                                    "expériences : grid2 = full + quads, bands = full + tb."},
                        "model": {"type": "string", "enum": list(MODELS),
                                  "description": "logreg (frontière droite) ou mlp (petit réseau, frontière "
                                                 "courbe). Aucun n'est privilégié."},
                        "params": {"type": "object",
                                   "description": "Frein contre l'apprentissage par cœur des chaînes : logreg C "
                                                  "(défaut 0.1, de 0.001 à 10, plus petit = plus freiné) ; mlp "
                                                  "alpha (défaut 1, de 0.0001 à 10, plus grand = plus freiné). "
                                                  "mlp seul : hidden, tailles des couches (défaut [256]). Les "
                                                  "deux : pca (int), classes équilibrées d'office ; le mlp "
                                                  "s'arrête sur des chaînes mises de côté."},
                        "balance": {"type": "object", "additionalProperties": {"type": "number"},
                                    "description": "{groupe: facteur de 0 à 10} sur les images d'entraînement : "
                                                   "sous 1 on en garde cette part, au-dessus on les duplique. "
                                                   "Groupes : " + ", ".join(BALANCE_GROUPS) + " (clip_vide : "
                                                   "vidéos sans aucun ajout)."},
                        "weights": {"type": "object", "additionalProperties": {"type": "number"},
                                    "description": "{groupe: poids de 0 à 10} dans la perte, mêmes groupes que "
                                                   "balance ; une image prend le plus grand poids de ses groupes."},
                        "note": {"type": "string", "description": "Hypothèse testée, en une phrase."},
                    },
                    "required": ["embeddings", "model"],
                },
                fn=lab.train_model,
            ),
            analyze_errors_tool(lab, session),
        ],
    )
