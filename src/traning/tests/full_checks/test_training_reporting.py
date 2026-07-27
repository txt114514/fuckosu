"""验证训练事件、资源快照和 UI reporter 的输出契约。"""

from __future__ import annotations

import unittest

from traning.lib import should_report_training_step


class TrainingReportingTests(unittest.TestCase):
    def test_step_reporting_is_throttled_and_keeps_boundaries(self) -> None:
        # 同时覆盖首步、热身步、普通节流点和最终步，避免节流优化吞掉
        # 生命周期边界事件。
        self.assertTrue(should_report_training_step(1, 100))
        self.assertTrue(should_report_training_step(5, 100))
        self.assertFalse(should_report_training_step(6, 100))
        self.assertTrue(should_report_training_step(100, 100))
        self.assertTrue(should_report_training_step(15, 300))
        self.assertFalse(should_report_training_step(16, 300))


if __name__ == "__main__":
    unittest.main()
