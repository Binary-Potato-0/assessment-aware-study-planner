"""Project entry point: train the NN separately, then run the full planner."""

# ENTRY POINT: delegate execution to the complete NN -> Search -> CSP demonstration.
from integration.demo import main


if __name__ == "__main__":
    main()
