"""Connect allocation search to CSP schedule feasibility.

The supplied base marks can come from a trained NN. Per-session gains remain
explicit scenario assumptions: OULAD does not identify their causal effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from csp import StudyScheduleCSP, format_schedule
from search import StudyAllocationProblem, search


@dataclass
class PlannerResult:
    success: bool
    algorithm: str
    allocation: tuple[int, ...]
    predicted_marks: tuple[float, ...]
    schedule: dict[str, tuple[int, int]]
    total_sessions: int
    nodes_expanded: int
    nodes_generated: int
    csp_assignments: int
    csp_backtracks: int
    reason: str


def plan(
    base_marks: Sequence[float],
    target_mark: float,
    deadline_weeks: Sequence[int],
    slots_per_week: int = 3,
    session_gain_assumptions: Sequence[float] | None = None,
    max_sessions_per_assignment: int = 8,
    algorithm: str = "ucs",
    use_forward_checking: bool = True,
    use_ac3: bool = False,
) -> PlannerResult:
    """Find a target-reaching allocation that also has a feasible timetable.

    Search calls the CSP validator whenever it reaches a target state. An
    unschedulable goal is rejected, and Search continues exploring.
    """
    base = tuple(float(x) for x in base_marks)
    deadlines = tuple(int(x) for x in deadline_weeks)
    if not base or len(base) != len(deadlines):
        raise ValueError("base_marks and deadline_weeks must have equal non-zero length")
    if any(not 0 <= x <= 100 for x in base):
        raise ValueError("base marks must be between 0 and 100")
    gains = (
        (5.0,) * len(base)
        if session_gain_assumptions is None
        else tuple(float(x) for x in session_gain_assumptions)
    )
    if len(gains) != len(base) or any(g <= 0 for g in gains):
        raise ValueError("one positive session-gain assumption is required per assignment")

    def scenario_predictor(allocation):
        return tuple(
            min(100.0, mark + gain * sessions)
            for mark, gain, sessions in zip(base, gains, allocation)
        )

    problem = StudyAllocationProblem(
        predictor=scenario_predictor,
        target_mark=target_mark,
        number_of_assignments=len(base),
        max_sessions_per_assignment=max_sessions_per_assignment,
        estimated_gain_per_session=gains,
    )
    csp_results = {}

    def schedule_validator(allocation):
        csp = StudyScheduleCSP(allocation, deadlines, slots_per_week)
        result = csp.solve(
            forward_checking=use_forward_checking,
            ac3=use_ac3,
        )
        csp_results[allocation] = result
        return result.success

    result = search(problem, algorithm=algorithm, goal_validator=schedule_validator)
    scheduled = csp_results.get(result.allocation)
    if not result.success or scheduled is None or not scheduled.success:
        return PlannerResult(
            False, algorithm, result.allocation, result.predicted_marks, {},
            result.total_sessions, result.nodes_expanded, result.nodes_generated,
            scheduled.assignments if scheduled else 0,
            scheduled.backtracks if scheduled else 0,
            result.reason,
        )

    return PlannerResult(
        True, algorithm, result.allocation, result.predicted_marks,
        format_schedule(scheduled.schedule, slots_per_week),
        result.total_sessions, result.nodes_expanded, result.nodes_generated,
        scheduled.assignments, scheduled.backtracks, scheduled.reason,
    )
