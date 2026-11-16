"""End-to-end demonstration of Search followed by CSP scheduling."""

from integration import plan


def main():
    result = plan(
        base_marks=(50, 52, 45),
        target_mark=65,
        deadline_weeks=(5, 4, 6),
        slots_per_week=3,
        session_gain_assumptions=(8, 5, 10),
        max_sessions_per_assignment=6,
        algorithm="ucs",
        use_forward_checking=True,
        use_ac3=True,
    )
    print("Assessment-aware Study Planner — integrated demonstration")
    print("Base predicted marks:", (50, 52, 45))
    print("Target:", 65)
    print("Allocation:", result.allocation)
    print("Predicted scenario marks:", tuple(round(x, 2) for x in result.predicted_marks))
    print("Total sessions:", result.total_sessions)
    print("Schedule (week, slot):", result.schedule)
    print("Search nodes expanded/generated:", result.nodes_expanded, result.nodes_generated)
    print("CSP assignments/backtracks:", result.csp_assignments, result.csp_backtracks)
    print("Success:", result.success)
    print("Reason:", result.reason)
    print("\nNOTE: Session gains are illustrative assumptions, not causal effects learned from OULAD.")


if __name__ == "__main__":
    main()
