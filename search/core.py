"""Search for the smallest study-session allocation that reaches a target.

A state is a tuple such as (1, 0, 2). It means:
- 1 session for assessment 1
- 0 sessions for assessment 2
- 2 sessions for assessment 3
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Callable, Literal, Sequence

Predictor = Callable[[tuple[int, ...]], Sequence[float]]
Algorithm = Literal["ucs", "greedy"]


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    success: bool
    allocation: tuple[int, ...]
    predicted_marks: tuple[float, ...]
    total_sessions: int
    nodes_expanded: int
    nodes_generated: int
    reason: str


class StudyAllocationProblem:
    """Defines the states, limits, predictions, and goal for the search."""

    def __init__(
        self,
        predictor: Predictor,
        target_mark: float,
        number_of_assignments: int,
        max_sessions_per_assignment: int | Sequence[int] = 8,
        estimated_gain_per_session: float | Sequence[float] = 5.0,
    ):
        if number_of_assignments < 1:
            raise ValueError("At least one assignment is required.")
        if not 0 <= target_mark <= 100:
            raise ValueError("Target mark must be between 0 and 100.")

        self.predictor = predictor
        self.target_mark = float(target_mark)
        self.number_of_assignments = number_of_assignments

        # Allow either one common limit or a separate limit per assignment.
        if isinstance(max_sessions_per_assignment, int):
            self.session_limits = (max_sessions_per_assignment,) * number_of_assignments
        else:
            self.session_limits = tuple(max_sessions_per_assignment)

        if len(self.session_limits) != number_of_assignments:
            raise ValueError("Provide one session limit per assignment.")
        if any(limit < 0 for limit in self.session_limits):
            raise ValueError("Session limits cannot be negative.")

        # The gains are planning assumptions, not effects learned by the NN.
        if isinstance(estimated_gain_per_session, (int, float)):
            self.session_gains = (float(estimated_gain_per_session),) * number_of_assignments
        else:
            self.session_gains = tuple(float(gain) for gain in estimated_gain_per_session)

        if len(self.session_gains) != number_of_assignments:
            raise ValueError("Provide one session gain per assignment.")
        if any(gain <= 0 for gain in self.session_gains):
            raise ValueError("Session gains must be positive.")

    def predict_marks(self, state: tuple[int, ...]) -> tuple[float, ...]:
        """Ask the supplied prediction function for marks in this state."""
        marks = tuple(float(mark) for mark in self.predictor(state))

        if len(marks) != self.number_of_assignments:
            raise ValueError("The predictor must return one mark per assignment.")
        if any(not math.isfinite(mark) for mark in marks):
            raise ValueError("Predicted marks must be finite numbers.")

        # Assessment marks cannot be below 0 or above 100.
        return tuple(max(0.0, min(100.0, mark)) for mark in marks)

    def reached_target(self, marks: tuple[float, ...]) -> bool:
        """The goal is reached only when every assessment meets the target."""
        return all(mark >= self.target_mark for mark in marks)

    def estimate_remaining_sessions(self, marks: tuple[float, ...]) -> float:
        """Estimate how many sessions are still needed (Greedy heuristic)."""
        estimate = 0.0
        for mark, gain in zip(marks, self.session_gains):
            deficit = max(0.0, self.target_mark - mark)
            estimate += deficit / gain
        return estimate

    def get_neighbors(self, state: tuple[int, ...]):
        """A neighbor adds one session to exactly one assignment."""
        for index, limit in enumerate(self.session_limits):
            if state[index] < limit:
                next_state = list(state)
                next_state[index] += 1
                yield tuple(next_state)


def search(
    problem: StudyAllocationProblem,
    algorithm: Algorithm = "ucs",
    goal_validator: Callable[[tuple[int, ...]], bool] | None = None,
) -> SearchResult:
    """Run Uniform-Cost Search or Greedy Best-First Search.

    UCS prioritizes the number of sessions already used, so it finds a
    minimum-session solution when a solution exists.

    Greedy prioritizes the estimated number of sessions still needed. It can
    be faster, but it does not guarantee the minimum-session solution.
    """
    if algorithm not in ("ucs", "greedy"):
        raise ValueError("Algorithm must be 'ucs' or 'greedy'.")

    start_state = (0,) * problem.number_of_assignments
    start_marks = problem.predict_marks(start_state)

    if problem.reached_target(start_marks):
        return SearchResult(
            algorithm, True, start_state, start_marks, 0, 0, 1,
            "Target already met without additional sessions.",
        )

    # The priority queue stores (priority, tie_breaker, state).
    # The counter prevents Python from comparing state tuples on a tie.
    counter = itertools.count()
    start_priority = 0.0
    if algorithm == "greedy":
        start_priority = problem.estimate_remaining_sessions(start_marks)

    frontier = [(start_priority, next(counter), start_state)]
    discovered = {start_state}
    nodes_expanded = 0
    nodes_generated = 1

    # Keep the best state seen in case no state reaches every target.
    best_state = start_state
    best_marks = start_marks

    def quality(marks, state):
        targets_met = sum(mark >= problem.target_mark for mark in marks)
        remaining_deficit = sum(
            max(0.0, problem.target_mark - mark) for mark in marks
        )
        return targets_met, -remaining_deficit, -sum(state)

    best_quality = quality(start_marks, start_state)

    while frontier:
        _, _, state = heapq.heappop(frontier)
        marks = problem.predict_marks(state)
        nodes_expanded += 1

        # The integration layer can reject a goal if its timetable is impossible.
        is_valid_goal = goal_validator is None or goal_validator(state)
        if problem.reached_target(marks) and is_valid_goal:
            return SearchResult(
                algorithm, True, state, marks, sum(state),
                nodes_expanded, nodes_generated,
                "Found an allocation meeting every target.",
            )

        current_quality = quality(marks, state)
        if current_quality > best_quality:
            best_state, best_marks = state, marks
            best_quality = current_quality

        for next_state in problem.get_neighbors(state):
            if next_state in discovered:
                continue

            discovered.add(next_state)
            nodes_generated += 1
            next_marks = problem.predict_marks(next_state)

            if algorithm == "ucs":
                priority = float(sum(next_state))
            else:
                priority = problem.estimate_remaining_sessions(next_marks)

            heapq.heappush(
                frontier, (priority, next(counter), next_state)
            )

    return SearchResult(
        algorithm, False, best_state, best_marks, sum(best_state),
        nodes_expanded, nodes_generated,
        "No feasible allocation meets the target within the session limits.",
    )
