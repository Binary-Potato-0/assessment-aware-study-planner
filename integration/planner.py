"""Combine the mark predictor, allocation search, and timetable CSP.

The neural network supplies baseline mark predictions.
Search chooses how many sessions each assessment receives.
The CSP checks whether those sessions fit before their deadlines.

Session gains are explicit scenario assumptions, not causal effects learned
from the OULAD dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from csp import StudyScheduleCSP, format_schedule
from search import StudyAllocationProblem, search


# ============================================================
# INTEGRATED RESULT: everything returned to the caller
# ============================================================
@dataclass
class PlannerResult:
    """All information returned by one planning attempt."""

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


# INPUT VALIDATION: reject invalid marks, deadlines, and gain assumptions.
def _validate_inputs(base_marks, target_mark, deadlines, gains):
    """Reject invalid input before starting the algorithms."""
    if not base_marks or len(base_marks) != len(deadlines):
        raise ValueError("Provide matching, non-empty marks and deadlines.")
    if not 0 <= target_mark <= 100:
        raise ValueError("Target mark must be between 0 and 100.")
    if any(mark < 0 or mark > 100 for mark in base_marks):
        raise ValueError("Baseline marks must be between 0 and 100.")
    if any(week < 0 for week in deadlines):
        raise ValueError("Deadlines cannot be negative.")
    if len(gains) != len(base_marks) or any(gain <= 0 for gain in gains):
        raise ValueError("Provide one positive session gain per assessment.")


# SCENARIO MARKS: apply the configured per-session assumptions.
def _scenario_mark_predictions(base_marks, gains, allocation):
    """Apply the assumed gain to each assessment, without exceeding 100."""
    return tuple(
        min(100.0, mark + gain * sessions)
        for mark, gain, sessions in zip(base_marks, gains, allocation)
    )


# ============================================================
# MAIN INTEGRATION: NN baseline -> Search allocation -> CSP timetable
# ============================================================
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
    """Find a target-reaching allocation that also has a valid timetable."""
    marks = tuple(float(mark) for mark in base_marks)
    deadlines = tuple(int(week) for week in deadline_weeks)

    if session_gain_assumptions is None:
        gains = (5.0,) * len(marks)
    else:
        gains = tuple(float(gain) for gain in session_gain_assumptions)

    _validate_inputs(marks, target_mark, deadlines, gains)

    # CASE 1: the student already meets every target, so zero sessions work.
    # If every baseline prediction already meets the target, no sessions
    # are required and an empty timetable is a valid solution.
    if all(mark >= target_mark for mark in marks):
        return PlannerResult(
            success=True,
            algorithm=algorithm,
            allocation=(0,) * len(marks),
            predicted_marks=marks,
            schedule={},
            total_sessions=0,
            nodes_expanded=0,
            nodes_generated=1,
            csp_assignments=0,
            csp_backtracks=0,
            reason="Target already met without additional sessions.",
        )

    # ADAPTER: give Search a mark-prediction function for each allocation.
    # Search needs a function that predicts marks for any proposed allocation.
    def predict_for_allocation(allocation):
        return _scenario_mark_predictions(marks, gains, allocation)

    problem = StudyAllocationProblem(
        predictor=predict_for_allocation,
        target_mark=target_mark,
        number_of_assignments=len(marks),
        max_sessions_per_assignment=max_sessions_per_assignment,
        estimated_gain_per_session=gains,
    )

    # CSP GOAL VALIDATOR: Search may find a mark allocation that cannot fit before the deadlines.
    # Store CSP results so we can return the timetable for the accepted goal.
    checked_schedules = {}

    def timetable_is_feasible(allocation):
        scheduler = StudyScheduleCSP(
            sessions_per_assignment=allocation,
            deadline_weeks=deadlines,
            slots_per_week=slots_per_week,
        )
        csp_result = scheduler.solve(
            forward_checking=use_forward_checking,
            ac3=use_ac3,
        )
        checked_schedules[allocation] = csp_result
        return csp_result.success

    # RUN SEARCH: the goal validator rejects allocations with impossible timetables.
    search_result = search(
        problem,
        algorithm=algorithm,
        goal_validator=timetable_is_feasible,
    )

    csp_result = checked_schedules.get(search_result.allocation)

    # FAILURE PATH: do not report success if no target-reaching allocation
    # also has a valid timetable.
    # No accepted allocation means Search exhausted its possible states.
    if not search_result.success or csp_result is None or not csp_result.success:
        return PlannerResult(
            success=False,
            algorithm=algorithm,
            allocation=search_result.allocation,
            predicted_marks=search_result.predicted_marks,
            schedule={},
            total_sessions=search_result.total_sessions,
            nodes_expanded=search_result.nodes_expanded,
            nodes_generated=search_result.nodes_generated,
            csp_assignments=csp_result.assignments if csp_result else 0,
            csp_backtracks=csp_result.backtracks if csp_result else 0,
            reason=search_result.reason,
        )

    # SUCCESS PATH: convert internal zero-based slots into readable week/slot pairs.
    # Convert internal zero-based slots into human-readable week/slot pairs.
    readable_schedule = format_schedule(
        csp_result.schedule,
        slots_per_week,
    )

    return PlannerResult(
        success=True,
        algorithm=algorithm,
        allocation=search_result.allocation,
        predicted_marks=search_result.predicted_marks,
        schedule=readable_schedule,
        total_sessions=search_result.total_sessions,
        nodes_expanded=search_result.nodes_expanded,
        nodes_generated=search_result.nodes_generated,
        csp_assignments=csp_result.assignments,
        csp_backtracks=csp_result.backtracks,
        reason=csp_result.reason,
    )
