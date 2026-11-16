"""End-to-end demonstration using the trained NN, Search and CSP."""

from nn.predict import predict_student_assignments
from integration import plan


def main():
    base_marks, student_id = predict_student_assignments()
    result = plan(
        base_marks=base_marks,
        target_mark=65,
        deadline_weeks=(5, 4, 6),
        slots_per_week=3,
        # Scenario assumptions only; not causal effects learned from OULAD.
        session_gain_assumptions=(8, 5, 10),
        max_sessions_per_assignment=8,
        algorithm="ucs",
        use_forward_checking=True,
        use_ac3=True,
    )
    print("Assessment-aware Study Planner — integrated demonstration")
    print("Held-out OULAD student:", student_id)
    print("NN predicted baseline marks:", tuple(round(x, 2) for x in base_marks))
    print("Target mark:", 65)
    print("Search allocation:", result.allocation)
    print("Scenario-adjusted marks:", tuple(round(x, 2) for x in result.predicted_marks))
    print("Total sessions:", result.total_sessions)
    print("CSP timetable (week, slot):", result.schedule)
    print("Search nodes expanded/generated:", result.nodes_expanded, result.nodes_generated)
    print("CSP assignments/backtracks:", result.csp_assignments, result.csp_backtracks)
    print("Success:", result.success)
    print("Reason:", result.reason)
    print("\nCAVEAT: Per-session gains are illustrative scenario assumptions,")
    print("not causal effects estimated from OULAD observations.")


if __name__ == "__main__":
    main()
