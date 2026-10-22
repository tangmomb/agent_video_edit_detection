import argparse

from .orchestrator import run


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m agents")
    p.add_argument("--goal", default="Lance une session d'amélioration du classifieur d'ajouts de montage.")
    p.add_argument("--budget-usd", type=float, default=1.00, help="budget API partagé par les 4 agents (en $)")
    p.add_argument("--data-budget", type=int, default=200, help="vidéos générables pendant la session")
    p.add_argument("--final-tests", type=int, default=2, help="appels à final_test autorisés")
    a = p.parse_args()
    run(a.goal, a.budget_usd, a.data_budget, a.final_tests)


if __name__ == "__main__":
    main()
