from __future__ import annotations

import unittest

from traning.lib import should_report_training_step


class TrainingReportingTests(unittest.TestCase):
    def test_step_reporting_is_throttled_and_keeps_boundaries(self) -> None:
        self.assertTrue(should_report_training_step(1, 100))
        self.assertTrue(should_report_training_step(5, 100))
        self.assertFalse(should_report_training_step(6, 100))
        self.assertTrue(should_report_training_step(100, 100))
        self.assertTrue(should_report_training_step(15, 300))
        self.assertFalse(should_report_training_step(16, 300))


if __name__ == "__main__":
    unittest.main()
