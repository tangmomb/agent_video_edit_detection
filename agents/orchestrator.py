"""Orchestrateur : décide de la prochaine expérience et délègue aux trois autres agents.

    python -m agents --budget-usd 1.00

Boucle : orchestrateur → (Dataset Builder | Model Researcher) → Evaluator → orchestrateur.
Chaque délégation lance le sous-agent sur une consigne autonome ; son rapport revient comme
résultat d'outil. Journaux dans runs/<horodatage>/.
"""

from model_training_pipeline.lab import Lab

from . import dataset_builder, evaluator, model_researcher
from .common import CONTEXT, dataset_info_tool, list_experiments_tool
from .runner import Agent, Session, Tool

SYSTEM = CONTEXT + """

Ton rôle : Orchestrateur. Tu pilotes l'amélioration du classifieur d'ajouts de montage en coordonnant trois \
agents, que tu sollicites avec les outils ask_* :
- Dataset Builder : génère des données synthétiques annotées ;
- Model Researcher : entraîne des modèles et mène des expériences sur val ;
- Evaluator : mesure, analyse les erreurs, vérifie les améliorations, accède au test et promeut \
en production.

Tu ne fais rien toi-même : tu lis l'état (status, list_experiments), tu choisis la prochaine \
expérience la plus utile, tu délègues, tu lis le rapport et tu décides de la suite. Cycle type : \
données si nécessaire → expériences → évaluation → décision. Les agents ne voient pas ta \
conversation : chaque consigne doit être autonome (objectif, contraintes, ids d'expériences \
concernés, ce qu'on attend dans le rapport). Ne demande des données que si l'analyse d'erreurs \
montre qu'elles manquent, en précisant lesquelles.

Les agents ne peuvent faire que ce que leurs outils permettent (liste plus bas) : ne leur demande \
jamais d'écrire ou d'exécuter un script, de lancer une commande ou de lire des fichiers. Chaque \
rapport se termine par une « Vérification automatique » produite par le code, qui liste les outils \
réellement appelés et leurs erreurs : elle fait foi. Toute affirmation du rapport qu'elle ne \
confirme pas (données générées, modèle entraîné, chiffres) est à considérer comme fausse. Relis \
status après chaque délégation plutôt que de croire le rapport sur parole.

Si une étape est impossible (dataset vide et budget de génération à 0, test final épuisé, etc.), \
ne la redemande pas sous une autre forme : rédige directement le rapport final en expliquant ce \
qui bloque.

Pour conclure, quand l'objectif est atteint, que les gains plafonnent ou que le budget l'impose : \
fais vérifier le meilleur candidat sur test par l'Evaluator et, si test confirme val, demande-lui \
de l'enregistrer en production. Puis rédige le rapport final en français : modèle retenu et \
pourquoi, chiffres val et test, déroulé des cycles, ce qui a marché ou non, recommandations."""

TASK = {"type": "object",
        "properties": {"task": {"type": "string", "description": "Consigne autonome pour l'agent."}},
        "required": ["task"]}


def make(lab: Lab, session: Session) -> Agent:
    agents = {"dataset_builder": dataset_builder.make(lab),
              "model_researcher": model_researcher.make(lab),
              "evaluator": evaluator.make(lab)}

    def delegate(name: str):
        return lambda task: session.run(agents[name], task)

    def status() -> dict:
        return {"dataset": lab.dataset_info(), "production": lab.production(),
                "budget_session_usd": {"depense": round(session.spent, 2), "total": session.budget_usd}}

    capabilities = "\n".join(f"- {name} : {', '.join(t.name for t in agent.tools)}"
                             for name, agent in agents.items())
    return Agent(
        name="orchestrator",
        system=SYSTEM + "\n\nOutils de chaque agent (rien d'autre n'est possible) :\n" + capabilities,
        max_turns=30,
        tools=[
            Tool(name="status",
                 description="État global : dataset, modèle en production, budget de session dépensé.",
                 input_schema={"type": "object", "properties": {}}, fn=status),
            list_experiments_tool(lab),
            Tool(name="ask_dataset_builder",
                 description="Délègue une génération de données au Dataset Builder ; renvoie son rapport.",
                 input_schema=TASK, fn=delegate("dataset_builder")),
            Tool(name="ask_model_researcher",
                 description="Délègue un lot d'expériences au Model Researcher ; renvoie son rapport.",
                 input_schema=TASK, fn=delegate("model_researcher")),
            Tool(name="ask_evaluator",
                 description="Délègue une évaluation, une analyse d'erreurs, un test final ou une mise en "
                             "production à l'Evaluator ; renvoie son rapport.",
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
    session.finish(report=report)
    (session.run_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"\n--- coût de la session : {session.spent:.2f} $ — journaux : {session.run_dir}")
    return session
