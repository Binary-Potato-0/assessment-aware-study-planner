"""Run the complete Neural Network -> Search -> CSP demonstration."""
from nn.predict import predict_student_assignments
from integration import plan


def main():
    # Step 1: Load the trained neural network and predict three assessment marks.
    base_marks, student_id = predict_student_assignments()

    # Step 2: Search for a session allocation that reaches the target.
    result = plan(
        base_marks=base_marks,
        target_mark=65,
        deadline_weeks=(5, 4, 6),
        slots_per_week=3,
        session_gain_assumptions=(8, 5, 10),
        max_sessions_per_assignment=8,
        algorithm="ucs",
        use_forward_checking=True,
        use_ac3=True,
    )

    # Step 3: Display the allocation and the timetable produced by the CSP.
    print("ASSESSMENT-AWARE STUDY PLANNER")
    print("=" * 40)
    print("Held-out student:", student_id)
    print("NN baseline predictions:", tuple(round(mark, 2) for mark in base_marks))
    print("Required target:", 65)
    print()
    print("Search allocation:", result.allocation)
    print("Scenario-adjusted marks:", tuple(round(mark, 2) for mark in result.predicted_marks))
    print("Total study sessions:", result.total_sessions)
    print("CSP timetable (week, slot):", result.schedule)
    print()
    print("Search nodes expanded:", result.nodes_expanded)
    print("Search nodes generated:", result.nodes_generated)
    print("CSP assignments:", result.csp_assignments)
    print("CSP backtracks:", result.csp_backtracks)
    print("Plan successful:", result.success)
    print("Explanation:", result.reason)
    print()
    print("NOTE: Session gains are illustrative planning assumptions.")
    print("They are not causal effects learned from OULAD data.")


if __name__ == "__main__":
    main()
