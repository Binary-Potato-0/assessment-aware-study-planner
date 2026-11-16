import unittest

from integration import plan


class IntegrationTests(unittest.TestCase):
    def test_pipeline_returns_feasible_plan(self):
        result = plan((50, 52, 45), 65, (5, 4, 6), 3, (8, 5, 10), 6, use_ac3=True)
        self.assertTrue(result.success)
        self.assertTrue(all(x >= 65 for x in result.predicted_marks))
        self.assertEqual(len(result.schedule), result.total_sessions)

    def test_unreachable_target_is_reported(self):
        result = plan((10, 10), 95, (3, 3), 2, (1, 1), 2)
        self.assertFalse(result.success)
        self.assertEqual(result.schedule, {})

    def test_impossible_deadline_is_reported(self):
        result = plan((40,), 60, (0,), 2, (10,), 3)
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
