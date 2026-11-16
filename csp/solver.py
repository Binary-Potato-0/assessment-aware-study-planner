"""Weekly study-session scheduling as a finite-domain CSP.

Each session is a variable. Values are integer weekly slot indices. Constraints:
all sessions use distinct slots, sessions precede their assignment deadline,
and sessions for the same assignment occur in increasing order.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable


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
            raise ValueError("session counts and deadlines must have equal length")
        if slots_per_week < 1 or any(n < 0 for n in sessions_per_assignment):
            raise ValueError("slots_per_week must be positive and sessions non-negative")
        if any(d < 0 for d in deadline_weeks):
            raise ValueError("deadlines must be non-negative week counts")
        self.slots_per_week = slots_per_week
        self.variables = []
        self.assignment_of = {}
        self.order_of = {}
        self.domains = {}
        for a, (count, deadline) in enumerate(zip(sessions_per_assignment, deadline_weeks)):
            names = []
            for j in range(count):
                name = f"A{a + 1}_S{j + 1}"
                names.append(name)
                self.variables.append(name)
                self.assignment_of[name] = a
                self.order_of[name] = j
                # Week 0 is the first available week; deadline is exclusive.
                self.domains[name] = set(range(int(deadline) * slots_per_week))
        self.predecessor = {}
        for a in range(len(sessions_per_assignment)):
            names = [v for v in self.variables if self.assignment_of[v] == a]
            for previous, current in zip(names, names[1:]):
                self.predecessor[current] = previous
        self.neighbors = {v: set() for v in self.variables}
        for i, x in enumerate(self.variables):
            for y in self.variables[i + 1:]:
                self.neighbors[x].add(y)
                self.neighbors[y].add(x)

    def compatible(self, x, vx, y, vy):
        if vx == vy:
            return False
        if self.predecessor.get(x) == y:
            return vy < vx
        if self.predecessor.get(y) == x:
            return vx < vy
        return True

    def solve(self, forward_checking=True, ac3=False):
        domains = {v: set(values) for v, values in self.domains.items()}
        if ac3 and not self._ac3(domains):
            return CSPResult(False, {}, 0, 0, "AC-3 detected an empty domain.")
        self.assignments = 0
        self.backtracks = 0
        solution = self._backtrack({}, domains, forward_checking, ac3)
        if solution is None:
            return CSPResult(
                False, {}, self.assignments, self.backtracks,
                "No schedule satisfies the deadline, capacity and ordering constraints.",
            )
        return CSPResult(True, solution, self.assignments, self.backtracks, "Schedule found.")

    def _select_variable(self, assignment, domains):
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda v: (len(domains[v]), v))

    def _backtrack(self, assignment, domains, forward_checking, ac3):
        if len(assignment) == len(self.variables):
            return dict(assignment)
        variable = self._select_variable(assignment, domains)
        for value in sorted(domains[variable]):
            if all(
                self.compatible(variable, value, other, other_value)
                for other, other_value in assignment.items()
            ):
                self.assignments += 1
                new_assignment = dict(assignment)
                new_assignment[variable] = value
                new_domains = {v: set(d) for v, d in domains.items()}
                new_domains[variable] = {value}
                if forward_checking:
                    consistent = self._forward_check(variable, value, new_assignment, new_domains)
                else:
                    consistent = True
                if consistent and ac3:
                    consistent = self._ac3(new_domains, assigned=new_assignment)
                if consistent:
                    result = self._backtrack(
                        new_assignment, new_domains, forward_checking, ac3
                    )
                    if result is not None:
                        return result
            self.backtracks += 1
        return None

    def _forward_check(self, variable, value, assignment, domains):
        for other in self.neighbors[variable]:
            if other in assignment:
                continue
            domains[other] = {
                candidate for candidate in domains[other]
                if self.compatible(variable, value, other, candidate)
            }
            if not domains[other]:
                return False
        return True

    def _ac3(self, domains, assigned=None):
        assigned = assigned or {}
        queue = deque(
            (x, y) for x in self.variables for y in self.neighbors[x]
            if x not in assigned and y not in assigned
        )
        while queue:
            x, y = queue.popleft()
            revised = {
                vx for vx in domains[x]
                if any(self.compatible(x, vx, y, vy) for vy in domains[y])
            }
            if revised != domains[x]:
                domains[x] = revised
                if not revised:
                    return False
                for z in self.neighbors[x] - {y}:
                    if z not in assigned:
                        queue.append((z, x))
        return all(domains[v] for v in self.variables if v not in assigned)


def format_schedule(schedule, slots_per_week):
    """Convert integer slot indices into human-readable week/slot pairs."""
    return {
        variable: (slot // slots_per_week + 1, slot % slots_per_week + 1)
        for variable, slot in sorted(schedule.items())
    }
