"""Constraint Satisfaction Problem (CSP) for weekly study scheduling.

Each study session is a variable. Its domain contains the available time slots
before that assessment's deadline.

Constraints:
1. Two sessions cannot occupy the same time slot.
2. Every session must happen before its assessment deadline.
3. Sessions for one assessment must stay in order.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class CSPResult:
    success: bool
    schedule: dict[str, int]
    assignments: int
    backtracks: int
    reason: str


class StudyScheduleCSP:
    def __init__(self, sessions_per_assignment, deadline_weeks, slots_per_week=3):
        if len(sessions_per_assignment) != len(deadline_weeks):
            raise ValueError("Session counts and deadlines must have equal length.")
        if slots_per_week < 1:
            raise ValueError("There must be at least one slot per week.")
        if any(count < 0 for count in sessions_per_assignment):
            raise ValueError("Session counts cannot be negative.")
        if any(week < 0 for week in deadline_weeks):
            raise ValueError("Deadlines cannot be negative.")

        self.slots_per_week = slots_per_week
        self.variables = []
        self.assignment_number = {}
        self.previous_session = {}
        self.domains = {}

        self._create_variables(sessions_per_assignment, deadline_weeks)
        self.neighbors = self._create_neighbors()

    def _create_variables(self, session_counts, deadlines):
        """Create one variable for every required study session."""
        for assignment_index, (count, deadline) in enumerate(
            zip(session_counts, deadlines), start=1
        ):
            assignment_variables = []

            for session_number in range(1, count + 1):
                variable = f"A{assignment_index}_S{session_number}"
                assignment_variables.append(variable)
                self.variables.append(variable)
                self.assignment_number[variable] = assignment_index

                # Integer slots start at zero. The deadline week is exclusive.
                last_slot = deadline * self.slots_per_week
                self.domains[variable] = set(range(last_slot))

            # Record which session must happen immediately before each session.
            for previous, current in zip(
                assignment_variables, assignment_variables[1:]
            ):
                self.previous_session[current] = previous

    def _create_neighbors(self):
        """Every pair of variables is connected by scheduling constraints."""
        neighbors = {variable: set() for variable in self.variables}

        for index, first in enumerate(self.variables):
            for second in self.variables[index + 1:]:
                neighbors[first].add(second)
                neighbors[second].add(first)

        return neighbors

    def is_compatible(self, first, first_slot, second, second_slot):
        """Check whether two sessions can use the proposed slots."""
        if first_slot == second_slot:
            return False

        # If first is the predecessor of second, first must be earlier.
        if self.previous_session.get(second) == first:
            return first_slot < second_slot

        # If second is the predecessor of first, second must be earlier.
        if self.previous_session.get(first) == second:
            return second_slot < first_slot

        return True

    def solve(self, forward_checking=True, ac3=False):
        """Try to find a complete schedule using backtracking."""
        domains = {
            variable: set(values)
            for variable, values in self.domains.items()
        }
        self.assignments = 0
        self.backtracks = 0

        if ac3 and not self._run_ac3(domains):
            return CSPResult(
                False, {}, self.assignments, self.backtracks,
                "AC-3 detected an empty domain.",
            )

        solution = self._backtrack(
            assignment={},
            domains=domains,
            forward_checking=forward_checking,
            use_ac3=ac3,
        )

        if solution is None:
            return CSPResult(
                False, {}, self.assignments, self.backtracks,
                "No schedule satisfies the deadline, capacity and ordering constraints.",
            )

        return CSPResult(
            True, solution, self.assignments, self.backtracks,
            "Schedule found.",
        )

    def _select_next_variable(self, assignment, domains):
        """MRV: choose the unassigned variable with the smallest domain."""
        unassigned = [
            variable for variable in self.variables
            if variable not in assignment
        ]
        return min(unassigned, key=lambda variable: len(domains[variable]))

    def _backtrack(self, assignment, domains, forward_checking, use_ac3):
        if len(assignment) == len(self.variables):
            return dict(assignment)

        variable = self._select_next_variable(assignment, domains)

        for slot in sorted(domains[variable]):
            # Check the candidate against all sessions already scheduled.
            fits_existing = all(
                self.is_compatible(variable, slot, other, other_slot)
                for other, other_slot in assignment.items()
            )
            if not fits_existing:
                self.backtracks += 1
                continue

            self.assignments += 1
            new_assignment = dict(assignment)
            new_assignment[variable] = slot

            # Copy domains so a failed branch does not affect another branch.
            new_domains = {
                name: set(values) for name, values in domains.items()
            }
            new_domains[variable] = {slot}

            consistent = True
            if forward_checking:
                consistent = self._forward_check(
                    variable, slot, new_assignment, new_domains
                )

            if consistent and use_ac3:
                consistent = self._run_ac3(
                    new_domains, assigned=new_assignment
                )

            if consistent:
                result = self._backtrack(
                    new_assignment, new_domains,
                    forward_checking, use_ac3,
                )
                if result is not None:
                    return result

            self.backtracks += 1

        return None

    def _forward_check(self, variable, slot, assignment, domains):
        """Remove values that conflict with the newly assigned session."""
        for neighbor in self.neighbors[variable]:
            if neighbor in assignment:
                continue

            domains[neighbor] = {
                candidate
                for candidate in domains[neighbor]
                if self.is_compatible(variable, slot, neighbor, candidate)
            }

            if not domains[neighbor]:
                return False

        return True

    def _run_ac3(self, domains, assigned=None):
        """Enforce arc consistency by revising neighboring variable domains."""
        assigned = assigned or {}
        queue = deque(
            (first, second)
            for first in self.variables
            for second in self.neighbors[first]
            if first not in assigned and second not in assigned
        )

        while queue:
            first, second = queue.popleft()
            supported_values = {
                first_slot
                for first_slot in domains[first]
                if any(
                    self.is_compatible(first, first_slot, second, second_slot)
                    for second_slot in domains[second]
                )
            }

            if supported_values == domains[first]:
                continue

            domains[first] = supported_values
            if not supported_values:
                return False

            for neighbor in self.neighbors[first] - {second}:
                if neighbor not in assigned:
                    queue.append((neighbor, first))

        return all(
            domains[variable]
            for variable in self.variables
            if variable not in assigned
        )


def format_schedule(schedule, slots_per_week):
    """Convert zero-based slot numbers to (week, slot), both starting at 1."""
    readable_schedule = {}

    for variable, slot in sorted(schedule.items()):
        week = slot // slots_per_week + 1
        slot_in_week = slot % slots_per_week + 1
        readable_schedule[variable] = (week, slot_in_week)

    return readable_schedule
