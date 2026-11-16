"""Uniform-Cost Search and Greedy Best-First Search for study allocation.

A state is a tuple where state[i] is the number of sessions allocated to
assignment i. The predictor is injected so search is independent of the
neural-network implementation.
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
    """Finite state-space model for allocating sessions to assignments."""

    def __init__(
        self,
        predictor: Predictor,
        target_mark: float,
        number_of_assignments: int,
        max_sessions_per_assignment: int | Sequence[int] = 8,
        estimated_gain_per_session: float | Sequence[float] = 5.0,
    ) -> None:
        if number_of_assignments < 1:
            raise ValueError("number_of_assignments must be at least 1")
        if not 0 <= target_mark <= 100:
            raise ValueError("target_mark must be between 0 and 100")

        self.predictor = predictor
        self.target_mark = float(target_mark)
        self.number_of_assignments = number_of_assignments

        if isinstance(max_sessions_per_assignment, int):
            limits = (max_sessions_per_assignment,) * number_of_assignments
        else:
            limits = tuple(int(x) for x in max_sessions_per_assignment)
        if len(limits) != number_of_assignments or any(x < 0 for x in limits):
            raise ValueError("session limits must be non-negative and match assignments")
        self.limits = limits

        if isinstance(estimated_gain_per_session, (int, float)):
            gains = (float(estimated_gain_per_session),) * number_of_assignments
        else:
            gains = tuple(float(x) for x in estimated_gain_per_session)
        if len(gains) != number_of_assignments or any(x <= 0 for x in gains):
            raise ValueError("estimated gains must be positive and match assignments")
        self.gains = gains

    def predict(self, state: tuple[int, ...]) -> tuple[float, ...]:
        marks = tuple(float(x) for x in self.predictor(state))
        if len(marks) != self.number_of_assignments:
            raise ValueError(
                f"predictor returned {len(marks)} marks for "
                f"{self.number_of_assignments} assignments"
            )
        if any(not math.isfinite(mark) for mark in marks):
            raise ValueError("predictor returned a non-finite mark")
        return tuple(max(0.0, min(100.0, mark)) for mark in marks)

    def is_goal(self, marks: tuple[float, ...]) -> bool:
        return all(mark >= self.target_mark for mark in marks)

    def heuristic(self, marks: tuple[float, ...]) -> float:
        """Estimate remaining sessions from mark deficits and gain assumptions."""
        return sum(
            max(0.0, self.target_mark - mark) / gain
            for mark, gain in zip(marks, self.gains)
        )

    def neighbors(self, state: tuple[int, ...]):
        """Generate states by adding exactly one session to one assignment."""
        for index, limit in enumerate(self.limits):
            if state[index] >= limit:
                continue
            child = list(state)
            child[index] += 1
            yield tuple(child)


def search(
    problem: StudyAllocationProblem,
    algorithm: Algorithm = "ucs",
    goal_validator: Callable[[tuple[int, ...]], bool] | None = None,
) -> SearchResult:
    """Run UCS or Greedy Best-First Search on the allocation state space.

    UCS prioritizes g(n), the number of sessions used, and is optimal because
    every action costs exactly one session. Greedy prioritizes h(n), the
    estimated remaining sessions, and is not guaranteed to be optimal.
    """
    if algorithm not in ("ucs", "greedy"):
        raise ValueError("algorithm must be 'ucs' or 'greedy'")

    start = (0,) * problem.number_of_assignments
    start_marks = problem.predict(start)
    if problem.is_goal(start_marks):
        return SearchResult(
            algorithm, True, start, start_marks, 0, 0, 1,
            "Target already met without additional sessions.",
        )

    counter = itertools.count()
    start_h = problem.heuristic(start_marks)
    start_priority = 0.0 if algorithm == "ucs" else start_h
    frontier = [(start_priority, next(counter), start)]
    discovered = {start}
    expanded = 0
    generated = 1

    best_state = start
    best_marks = start_marks
    best_key = (
        sum(mark >= problem.target_mark for mark in start_marks),
        -sum(max(0.0, problem.target_mark - mark) for mark in start_marks),
        -sum(start),
    )

    while frontier:
        _, _, state = heapq.heappop(frontier)
        marks = problem.predict(state)
        expanded += 1

        if problem.is_goal(marks) and (goal_validator is None or goal_validator(state)):
            return SearchResult(
                algorithm, True, state, marks, sum(state), expanded, generated,
                "Found an allocation meeting every target.",
            )

        current_key = (
            sum(mark >= problem.target_mark for mark in marks),
            -sum(max(0.0, problem.target_mark - mark) for mark in marks),
            -sum(state),
        )
        if current_key > best_key:
            best_state, best_marks, best_key = state, marks, current_key

        for child in problem.neighbors(state):
            if child in discovered:
                continue
            discovered.add(child)
            generated += 1
            child_marks = problem.predict(child)
            if algorithm == "ucs":
                priority = float(sum(child))
            else:
                priority = problem.heuristic(child_marks)
            heapq.heappush(frontier, (priority, next(counter), child))

    return SearchResult(
        algorithm, False, best_state, best_marks, sum(best_state),
        expanded, generated,
        "No allocation meets the target and passes the feasibility validator within the session limits.",
    )
