"""Orchestrateur : décide de la prochaine expérience et délègue aux trois autres agents.

    python -m agents --budget-usd 1.00

Boucle : orchestrateur → (Dataset Builder | Model Researcher) → Evaluator → orchestrateur.
Chaque délégation lance le sous-agent sur une consigne autonome ; son rapport revient comme
résultat d'outil. Journaux dans runs/<NNN>_<horodatage>/.
"""

from model_training_pipeline.lab import Lab

from . import dataset_builder, evaluator, model_researcher
from .common import CONTEXT, list_experiments_tool
from .runner import Agent, Session, Tool

SYSTEM = CONTEXT + """

Ton rôle : Orchestrateur. Tu pilotes l'amélioration du classifieur d'ajouts de montage en coordonnant trois \
agents, que tu sollicites avec les outils ask_* :
- Dataset Builder : génère des données synthétiques annotées, sans se soucier des splits ;
- Model Researcher : constitue les splits train/val/test, entraîne des modèles et mène des \
expériences sur val ;
- Evaluator : mesure, analyse les erreurs, vérifie les améliorations et accède au test.

Tu ne fais rien toi-même : tu lis l'état (global_status, list_experiments), tu choisis la prochaine \
expérience la plus utile, tu délègues, tu lis le rapport et tu décides de la suite. Cycle type : \
données si nécessaire → split si le dataset a changé → expériences → évaluation → décision. Les agents ne voient pas ta \
conversation : chaque consigne doit être autonome (objectif, contraintes, ids d'expériences \
concernés, ce qu'on attend dans le rapport). Ne demande des données que si l'analyse d'erreurs \
montre qu'elles manquent, en précisant lesquelles.

Les agents ne peuvent faire que ce que leurs outils permettent (liste plus bas) : ne leur demande \
jamais d'écrire ou d'exécuter un script, de lancer une commande ou de lire des fichiers. Chaque \
rapport se termine par une « Vérification automatique » produite par le code, qui liste les outils \
réellement appelés et leurs erreurs : elle fait foi. Toute affirmation du rapport qu'elle ne \
confirme pas (données générées, modèle entraîné, chiffres) est à considérer comme fausse. Relis \
global_status après chaque délégation plutôt que de croire le rapport sur parole.

Si une étape est impossible (dataset vide et budget de génération à 0, test final épuisé, etc.), \
ne la redemande pas sous une autre forme : rédige directement le rapport final en expliquant ce \
qui bloque.

Budget : une part du budget de session (reserve_rapport_usd dans global_status) t'est réservée \
pour rédiger le rapport final. Une fois le reste dépensé, les délégations sont refusées : planifie \
le test final avant. Un sous-agent à court de budget rend quand même un \
rapport, signalé comme tel.

Pour conclure, quand l'objectif est atteint, que les gains plafonnent ou que le budget l'impose : \
fais vérifier le meilleur candidat sur test par l'Evaluator. Aucun agent ne met de modèle en \
production : à la fin du run, le code y met l'expérience de meilleur rappel val à précision 95 % \
sur le plus gros split (le plus de vidéos), et l'ajoute à ton rapport. Pour que ce choix soit le \
bon, fais entraîner les meilleures configs sur le split le plus récent. Rédige ton rapport en \
français : modèle recommandé (id d'expérience, seuil, split) et pourquoi, verdict du test, chiffres val et test, déroulé des cycles, ce qui a marché ou non, recommandations."""

TASK = {"type": "object",
        "properties": {"task": {"type": "string", "description": "Consigne autonome pour l'agent."}},
        "required": ["task"]}


def make(lab: Lab, session: Session) -> Agent:
    agents = {"dataset_builder": dataset_builder.make(lab),
              "model_researcher": model_researcher.make(lab),
              "evaluator": evaluator.make(lab)}

    def delegate(name: str):
        def ask(task: str):
            if session.reserve_reached:
                return {"error": f"budget de délégation épuisé ({session.spent:.2f} $ sur "
                                 f"{session.budget_usd:.2f} $, dont {session.reserve_usd:.2f} $ réservés à ton "
                                 "rapport) : rédige maintenant ton rapport final."}
            return session.run(agents[name], task)
        return ask

    def global_status() -> dict:
        return {"dataset": lab.dataset_info(),
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
                 description="État global : dataset (composition globale : vidéos, frames, taux de "
                             "positifs, répartition par orientation, type d'élément, animations d'entrée et "
                             "de sortie, surface, fond ; splits existants ; diversité ; budgets), budget de session "
                             "dépensé.",
                 input_schema={"type": "object", "properties": {}}, fn=global_status),
            list_experiments_tool(lab),
            Tool(name="ask_dataset_builder",
                 description="Délègue une génération de données au Dataset Builder ; renvoie son rapport.",
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


def run(goal: str, budget_usd: float, data_budget: int, final_tests: int, client=None) -> Session:
    lab = Lab(data_budget=data_budget, final_test_budget=final_tests)
    session = Session(budget_usd, client=client,
                      params={"goal": goal, "data_budget": data_budget, "final_tests": final_tests})
    try:
        report = session.run(make(lab, session), goal)
    except BaseException as e:
        session.finish(error=e)
        raise
    # Le modèle de production est choisi par le code, pas par un agent (lab.select_production).
    choice = lab.select_production()
    report += "\n\n---\n\n## Modèle de production (choix automatique du code)\n\n" + (
        f"Erreur : {choice['error']}" if "error" in choice else
        f"`{choice['production']['experiment_id']}` sur `{choice['split_id']}` (le plus gros split), seuil "
        f"{choice['production']['threshold']} : meilleur rappel val à précision "
        f"{choice['production']['precision_visee']}. Classement :\n\n"
        + "\n".join(f"- {r['experiment_id']} : rappel val {r['val_recall']}, AP {r['ap']}, AUC {r['auc']}, "
                    f"{r['ms_par_frame']} ms/frame" for r in choice["classement"]))
    session.finish(report=report)
    (session.run_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"\n--- coût de la session : {session.spent:.2f} $ — journaux : {session.run_dir}")
    return session
