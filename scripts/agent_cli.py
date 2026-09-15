import os
import sys
import json


# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


from src.agent_core import run_agent


def main():

    print(
        "AI Media Task Agent (type 'exit' to quit)\n"
    )

    while True:

        goal = input(
            "Goal: "
        ).strip()

        if goal.lower() in {
            "exit",
            "quit"
        }:
            break

        if not goal:

            print(
                "Please enter a goal.\n"
            )

            continue

        result = run_agent(
            goal,
            max_steps=8
        )

        print(
            "\nResult:"
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )

        print()


if __name__ == "__main__":
    main()