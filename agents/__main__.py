import argparse

from model_training_pipeline import batch

from .orchestrator import run


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m agents")
    p.add_argument("--goal", default="Lance une session d'amélioration du classifieur d'ajouts de montage.")
    p.add_argument("--budget-usd", type=float, default=1.00, help="budget API partagé par les 4 agents (en $)")
    p.add_argument("--data-budget", type=int, default=250, help="vidéos YouTube collectables pendant la session")
    p.add_argument("--annotation-usd", type=float, default=2.00,
                   help="budget d'annotation des images par les 3 LLM (en $, en plus de --budget-usd)")
    p.add_argument("--final-tests", type=int, default=2, help="appels à final_test autorisés")
    p.add_argument("--agent-model",
                   help="même modèle pour les 4 agents, au format LiteLLM « fournisseur/identifiant » "
                        "(ex. openai/gpt-5.6-luna, anthropic/claude-haiku-5-5) ; défaut : mistral/mistral-large-4-0")
    p.add_argument("--direct-annotation", action="store_true",
                   help="annotation en appels directs (prix plein, sans attente) au lieu des batchs (-50 %%)")
    p.add_argument("--direct-agents", action="store_true",
                   help="agents en appels directs (prix plein, sans attente) au lieu des batchs Mistral (-50 %%)")
    a = p.parse_args()
    # Des votes encore dans des batchs : le dataset que verraient les agents serait incomplet.
    pending = batch.in_flight()
    if pending:
        raise SystemExit(f"{len(pending)} batch(s) d'annotation pas encore récupéré(s) : lance d'abord "
                         "`python -m model_training_pipeline batch collect` (batch status pour suivre).")
    run(a.goal, a.budget_usd, a.data_budget, a.final_tests, a.annotation_usd, model=a.agent_model,
        direct_annotation=a.direct_annotation, direct_agents=a.direct_agents)


if __name__ == "__main__":
    main()
