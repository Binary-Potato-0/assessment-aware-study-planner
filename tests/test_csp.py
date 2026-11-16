import unittest

from csp import StudyScheduleCSP


class CSPTests(unittest.TestCase):
    def test_backtracking_finds_schedule(self):
        result = StudyScheduleCSP([2, 1], [3, 2], 2).solve()
        self.assertTrue(result.success)
        self.assertEqual(len(set(result.schedule.values())), 3)

    def test_forward_checking_finds_schedule(self):
        result = StudyScheduleCSP([2, 2], [3, 3], 2).solve(forward_checking=True)
        self.assertTrue(result.success)

    def test_ac3_finds_schedule(self):
        result = StudyScheduleCSP([2, 2], [3, 3], 2).solve(
            forward_checking=True, ac3=True
        )
        self.assertTrue(result.success)

    def test_impossible_due_to_insufficient_slots(self):
        result = StudyScheduleCSP([3], [1], 2).solve()
        self.assertFalse(result.success)
        self.assertEqual(result.schedule, {})

    def test_deadline_zero_with_required_session_is_impossible(self):
        result = StudyScheduleCSP([1], [0], 3).solve(ac3=True)
        self.assertFalse(result.success)

    def test_zero_sessions_is_valid(self):
        result = StudyScheduleCSP([0, 0], [0, 0], 2).solve()
        self.assertTrue(result.success)
        self.assertEqual(result.schedule, {})

    def test_sessions_for_same_assignment_are_ordered(self):
        problem = StudyScheduleCSP([3], [3], 2)
        result = problem.solve()
        self.assertTrue(result.success)
        slots = [result.schedule[f"A1_S{i}"] for i in (1, 2, 3)]
        self.assertEqual(slots, sorted(slots))


if __name__ == "__main__":
    unittest.main()
