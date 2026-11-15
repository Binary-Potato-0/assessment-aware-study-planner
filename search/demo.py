"""Run a small deterministic example of the study-allocation search."""

from search import StudyAllocationProblem, search


BASE_MARKS = (50.0, 52.0, 45.0)
SCENARIO_GAIN = (8.0, 5.0, 10.0)
TARGET = 65.0


def illustrative_predictor(allocation: tuple[int, ...]) -> tuple[float, ...]:
    """Transparent toy scenario; not a causal model trained on OULAD."""
    return tuple(
        min(100.0, base + gain * sessions)
        for base, gain, sessions in zip(BASE_MARKS, SCENARIO_GAIN, allocation)
    )


def main() -> None:
    problem = StudyAllocationProblem(
        predictor=illustrative_predictor,
        target_mark=TARGET,
        number_of_assignments=3,
        max_sessions_per_assignment=6,
        estimated_gain_per_session=SCENARIO_GAIN,
    )

    print("Assessment-aware Study Planner — Search demonstration")
    print(f"Starting predicted marks: {BASE_MARKS}")
    print(f"Target mark: {TARGET:.1f}")
    print("Illustrative gain assumptions per session:", SCENARIO_GAIN)

    for algorithm in ("ucs", "greedy"):
        result = search(problem, algorithm=algorithm)
        print(f"\n{algorithm.upper()}")
        print("Success:", result.success)
        print("Sessions per assignment:", result.allocation)
        print("Predicted marks:", tuple(round(x, 2) for x in result.predicted_marks))
        print("Total sessions:", result.total_sessions)
        print("Nodes expanded:", result.nodes_expanded)
        print("Nodes generated:", result.nodes_generated)
        print("Reason:", result.reason)


if __name__ == "__main__":
    main()
