"""Orchestrateur : décide de la prochaine expérience et délègue aux trois autres agents.

    python -m agents --budget-usd 1.00

Boucle : orchestrateur → (Dataset Builder | Model Researcher) → Evaluator → orchestrateur.
Chaque délégation lance le sous-agent sur une consigne autonome ; son rapport revient comme
résultat d'outil. Journaux dans runs/<NNN>_<horodatage>/.
"""

from model_training_pipeline.lab import AP_TIE, Lab

from . import dataset_builder, evaluator, model_researcher
from .common import CONTEXT, MEASURE, list_experiments_tool
from .runner import Agent, Session, Tool

SYSTEM = CONTEXT + MEASURE + """

Ton rôle : Orchestrateur. Tu ne fais rien toi-même : tu lis l'état (global_status, \
list_experiments), tu délègues à trois agents avec les outils ask_* et tu décides de la suite.
- Dataset Builder : collecte de vidéos.
- Model Researcher : splits et entraînements, mesurés sur val.
- Evaluator : analyse des erreurs, seul accès au test.
Cycle type : données si besoin → nouveau split → expériences → évaluation → décision.

Ton but : un modèle en production entraîné sur le plus de données possible, et le meilleur sur ces \
données. Commence par global_status et décide :
- s'il reste des vidéos annotables, fais-les collecter : plus de données, c'est le levier principal ;
- si le modèle en production vient d'un split qui a moins de vidéos que celles disponibles \
(production.videos_du_split < videos_disponibles), ou s'il n'y en a pas, il faut un nouveau split, \
des expériences et un test final ;
- sinon, il n'y a rien à faire : conclus tout de suite, sans déléguer, en le disant.

Délégation :
- Les agents ne voient pas ta conversation : chaque consigne est autonome (objectif, contraintes, \
ids d'expériences, contenu attendu du rapport).
- Ne leur demande que ce que leurs outils permettent (liste en bas) : jamais de script, de \
commande ou de fichier.
- Chaque rapport finit par une « Vérification automatique » qui liste les outils réellement \
appelés : elle fait foi, et ce qu'elle ne confirme pas est faux. Relis global_status après chaque \
délégation.
- Une étape impossible (budget à 0, test final épuisé…) : ne la redemande pas autrement, conclus.

Données : avant une collecte, fais analyser les erreurs du meilleur modèle et donne au Dataset \
Builder les types de montage à viser, tels que l'analyse les décrit (ex. « sous-titres blancs en \
bas », « petit logo en coin »). Fais ensuite créer un nouveau split et réentraîner la meilleure \
config dessus.

Budget : une réserve (reserve_rapport_usd) est gardée pour ton rapport ; une fois le reste \
dépensé, les délégations sont refusées. Planifie le test final avant.

Pour conclure (gains qui plafonnent ou budget) : fais tester par l'Evaluator candidat_a_tester \
(list_experiments). S'il échoue et qu'il reste un test, le suivant. Aucun agent ne met de modèle \
en production : le code \
choisit ensuite le meilleur parmi les modèles qui ont réussi leur test (à défaut, le meilleur, \
marqué non vérifié) et l'ajoute à ton rapport. Un test est réussi dès que la précision visée tient \
sur test, même si le rappel baisse. N'annonce pas quel modèle sera en production : c'est le code \
qui le décide, après ton rapport (run 010 : rapport « exp_045 reste en production », code : exp_151).

Rapport final, en français : modèle recommandé (id, seuil, split) et pourquoi, verdict du test, \
chiffres val et test, déroulé, ce qui a marché ou non, recommandations. Le gain de la session se \
mesure contre la meilleure config d'avant réentraînée sur le split final."""

TASK = {"type": "object",
        "properties": {"task": {"type": "string", "description": "Consigne autonome pour l'agent."}},
        "required": ["task"]}


def make(lab: Lab, session: Session) -> Agent:
    agents = {"dataset_builder": dataset_builder.make(lab),
              "model_researcher": model_researcher.make(lab, session),
              "evaluator": evaluator.make(lab, session)}

    def delegate(name: str):
        def ask(task: str):
            if session.reserve_reached:
                return {"error": f"budget de délégation épuisé ({session.spent:.2f} $ sur "
                                 f"{session.budget_usd:.2f} $, dont {session.reserve_usd:.2f} $ réservés à ton "
                                 "rapport) : rédige maintenant ton rapport final."}
            return session.run(agents[name], task)
        return ask

    def global_status() -> dict:
        # Volumes seulement : le détail de composition (orientation, chaînes)
        # sert aux sous-agents, qui le lisent par leur propre dataset_info.
        info = lab.dataset_info()
        if info.get("dataset_vide"):
            dataset = info
        else:
            d = info["dataset_total_disponible"]
            dataset = {k: d[k] for k in ("chaines_distinctes", "videos", "moyenne_videos_par_chaine",
                                         "images_etiquetees", "videos_sans_ajout",
                                         "pourcentage_images_avec_ajout")}
            dataset["budgets"] = info["budgets"]
        return {"dataset_total_disponible": dataset, "production": lab.production_status(),
                "budget_session_usd": {"depense": round(session.spent, 2), "total": session.budget_usd,
                                       "reserve_rapport_usd": round(session.reserve_usd, 2),
                                       "disponible_pour_deleguer": round(max(
                                           0.0, session.budget_usd - session.reserve_usd - session.spent), 2)}}

    capabilities = "\n".join(f"- {name} : {', '.join(t.name for t in agent.tools)}"
                             for name, agent in agents.items())
    return Agent(
        name="orchestrator",
        system=SYSTEM + "\n\nOutils de chaque agent (rien d'autre n'est possible) :\n" + capabilities,
        max_turns=30,
        tools=[
            Tool(name="global_status",
                 description="Volumes du dataset, budgets restants (vidéos, annotation, tests finaux), "
                             "modèle en production (son split, son test, les vidéos de son split contre "
                             "celles disponibles aujourd'hui) et budget de session.",
                 input_schema={"type": "object", "properties": {}}, fn=global_status),
            list_experiments_tool(lab, brief=True),
            Tool(name="ask_dataset_builder",
                 description="Délègue une collecte de données au Dataset Builder ; renvoie son rapport.",
                 input_schema=TASK, fn=delegate("dataset_builder")),
            Tool(name="ask_model_researcher",
                 description="Délègue un lot d'expériences au Model Researcher ; renvoie son rapport.",
                 input_schema=TASK, fn=delegate("model_researcher")),
            Tool(name="ask_evaluator",
                 description="Délègue une évaluation, une analyse d'erreurs ou un test final à "
                             "l'Evaluator ; renvoie son rapport.",
                 input_schema=TASK, fn=delegate("evaluator")),
        ],
    )


def run(goal: str, budget_usd: float, data_budget: int, final_tests: int, annotation_usd: float = 2.0,
        model: str | None = None, completion=None, direct_annotation: bool = False,
        direct_agents: bool = False) -> Session:
    lab = Lab(data_budget=data_budget, final_test_budget=final_tests, annotation_budget_usd=annotation_usd,
              direct_annotation=direct_annotation)
    session = Session(budget_usd, completion=completion, model=model, batch_agents=not direct_agents,
                      params={"goal": goal, "data_budget": data_budget, "final_tests": final_tests,
                              "annotation_usd": annotation_usd, "agent_model": model,
                              "annotation_directe": direct_annotation, "agents_directs": direct_agents})
    try:
        report = session.run(make(lab, session), goal)
    except BaseException as e:
        session.finish(error=e)
        raise
    # Le modèle de production est choisi par le code, pas par un agent (lab.select_production).
    choice = lab.select_production()
    report += "\n\n---\n\n## Modèle de production (choix automatique du code)\n\n" + (
        f"Erreur : {choice['error']}" if "error" in choice else
        f"`{choice['production']['experiment_id']}` sur `{choice['production']['split_id']}` conservé : "
        f"il a réussi son test final, et aucun modèle de `{choice['split_id']}` (le plus gros split) "
        "n'a réussi le sien. Classement de ce split :\n\n"
        + "\n".join(f"- {r['experiment_id']} : AP par chaîne {r['ap_par_chaine']}, rappel val "
                    f"{r['val_recall']}, AP {r['ap']}, AUC {r['auc']}, "
                    f"test {r['test']['statut']}" for r in choice["classement"])
        if choice["production_conservee"] else
        f"`{choice['production']['experiment_id']}` sur `{choice['split_id']}` (le plus gros split), seuil "
        f"{choice['production']['threshold']} : "
        + (f"meilleur rappel val parmi les AP par chaîne à {AP_TIE} près de la meilleure, chez les "
           "modèles qui ont réussi leur test final."
           if choice["verifie_sur_test"] else
           "**NON VÉRIFIÉ sur test** : aucun modèle de ce split n'a réussi son test final, c'est le "
           f"meilleur rappel val parmi les AP par chaîne à {AP_TIE} près de la meilleure, chez les modèles "
           + ("non testés." if choice["production"]["test"]["statut"] == "non testé"
              else "testés, tous en échec."))
        + " Classement :\n\n"
        + "\n".join(f"- {r['experiment_id']} : AP par chaîne {r['ap_par_chaine']}, rappel val "
                    f"{r['val_recall']}, AP {r['ap']}, AUC {r['auc']}, "
                    f"test {r['test']['statut']}" for r in choice["classement"]))
    session.finish(report=report)
    (session.run_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"\n--- coût de la session : {session.spent:.2f} $ — journaux : {session.run_dir}")
    return session
