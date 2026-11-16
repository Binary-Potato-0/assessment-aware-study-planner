"""Connect session-allocation search to weekly CSP scheduling.

The supplied base-mark predictor may be produced by the trained NN. Additional
session gains are explicit scenario assumptions, not causal estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Callable

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
    """Find a minimum-session target allocation and a feasible timetable.

    Search first proposes allocations in increasing cost order (UCS). If a
    candidate cannot be scheduled, it is rejected and search continues.
    Greedy is available for comparison, but does not guarantee minimum cost.
    """
    base = tuple(float(x) for x in base_marks)
    deadlines = tuple(int(x) for x in deadline_weeks)
    if not base or len(base) != len(deadlines):
        raise ValueError("base_marks and deadline_weeks must have equal non-zero length")
    if any(not 0 <= x <= 100 for x in base):
        raise ValueError("base marks must be between 0 and 100")
    if session_gain_assumptions is None:
        gains = (5.0,) * len(base)
    else:
        gains = tuple(float(x) for x in session_gain_assumptions)
    if len(gains) != len(base):
        raise ValueError("one session-gain assumption is required per assignment")

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

    # Search returns a target-reaching allocation. Scheduling it is a separate
    # feasibility check; if it fails, this first implementation reports that
    # failure rather than silently claiming a timetable exists.
    result = search(problem, algorithm=algorithm)
    if not result.success:
        return PlannerResult(
            False, algorithm, result.allocation, result.predicted_marks, {},
            result.total_sessions, result.nodes_expanded, result.nodes_generated,
            0, 0, result.reason,
        )

    csp = StudyScheduleCSP(result.allocation, deadlines, slots_per_week)
    scheduled = csp.solve(
        forward_checking=use_forward_checking,
        ac3=use_ac3,
    )
    return PlannerResult(
        scheduled.success, algorithm, result.allocation, result.predicted_marks,
        format_schedule(scheduled.schedule, slots_per_week) if scheduled.success else {},
        result.total_sessions, result.nodes_expanded, result.nodes_generated,
        scheduled.assignments, scheduled.backtracks, scheduled.reason,
    )
