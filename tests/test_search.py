import unittest

from search import StudyAllocationProblem, search


class SearchTests(unittest.TestCase):
    def make_problem(self, bases, gains, target, limits=6):
        def predictor(state):
            return tuple(
                min(100.0, base + gain * sessions)
                for base, gain, sessions in zip(bases, gains, state)
            )

        return StudyAllocationProblem(
            predictor, target, len(bases), limits, gains
        )

    def test_ucs_finds_minimum_session_allocation(self):
        problem = self.make_problem((50, 52, 45), (8, 5, 10), 65)
        result = search(problem, "ucs")
        self.assertTrue(result.success)
        self.assertEqual(result.total_sessions, 7)
        self.assertEqual(result.allocation, (2, 3, 2))

    def test_greedy_finds_a_goal(self):
        problem = self.make_problem((50, 52, 45), (8, 5, 10), 65)
        result = search(problem, "greedy")
        self.assertTrue(result.success)
        self.assertTrue(all(mark >= 65 for mark in result.predicted_marks))

    def test_already_satisfied_requires_zero_sessions(self):
        problem = self.make_problem((80, 75), (5, 5), 70)
        result = search(problem, "ucs")
        self.assertTrue(result.success)
        self.assertEqual(result.total_sessions, 0)
        self.assertEqual(result.nodes_expanded, 0)

    def test_impossible_target_returns_best_effort(self):
        problem = self.make_problem((20, 30), (2, 2), 90, limits=2)
        result = search(problem, "ucs")
        self.assertFalse(result.success)
        self.assertEqual(result.total_sessions, 4)
        self.assertIn("No feasible allocation", result.reason)

    def test_zero_session_limits_can_make_problem_impossible(self):
        problem = self.make_problem((40, 40), (5, 5), 60, limits=0)
        result = search(problem, "ucs")
        self.assertFalse(result.success)
        self.assertEqual(result.total_sessions, 0)
        self.assertEqual(result.nodes_generated, 1)

    def test_predictor_must_return_one_mark_per_assignment(self):
        problem = StudyAllocationProblem(lambda state: (50,), 60, 2)
        with self.assertRaises(ValueError):
            search(problem, "ucs")

    def test_invalid_algorithm_is_rejected(self):
        problem = self.make_problem((40,), (5,), 60)
        with self.assertRaises(ValueError):
            search(problem, "astar")

    def test_search_counters_are_consistent(self):
        problem = self.make_problem((50,), (10,), 70)
        result = search(problem, "ucs")
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.nodes_generated, result.nodes_expanded)
        self.assertEqual(result.total_sessions, 2)


if __name__ == "__main__":
    unittest.main()
