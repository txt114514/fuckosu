"""验证 trial 聚合分不会被大量正确 no-op 空帧虚高。"""

from __future__ import annotations

import unittest

from traning.core.optimization.scoring.evaluator import (
    AGGREGATE_SCORE_VERSION,
    SampleScoringInput,
    score_trial,
)
from traning.lib.metrics import TargetObject


class TrialAggregateScoringTests(unittest.TestCase):
    def test_background_only_trial_cannot_pass_without_target_coverage(self) -> None:
        samples = tuple(
            SampleScoringInput(
                sample_key=f"background-only/{index:04d}",
                subproject="long_sequence",
                targets=(),
                predictions=(),
                circle_radius=32.0,
                frame_index=index,
            )
            for index in range(8)
        )

        report = score_trial("background-only", samples)

        self.assertEqual(report.target_count, 0)
        self.assertEqual(report.quality_score, 1.0)
        self.assertTrue(all(sample.passed for sample in report.samples))
        self.assertFalse(report.passed)

    def test_no_op_background_frames_cannot_hide_all_unresolved_targets(self) -> None:
        # 复现 20260727 运行的结构：412 个背景帧正确 no-op，88 个目标帧
        # 没有任何点击。旧版直接按 500 帧平均得到 0.824，误导参数搜索。
        background = tuple(
            SampleScoringInput(
                sample_key=f"background/{index:04d}",
                subproject="long_sequence",
                targets=(),
                predictions=(),
                circle_radius=32.0,
                frame_index=index,
            )
            for index in range(412)
        )
        unresolved = tuple(
            SampleScoringInput(
                sample_key=f"target/{index:04d}",
                subproject="long_sequence",
                targets=(
                    TargetObject(
                        target_id=f"target-{index}",
                        target_type="circle",
                        start_ms=1000.0,
                        end_ms=1000.0,
                        x=100.0,
                        y=100.0,
                    ),
                ),
                predictions=(),
                circle_radius=32.0,
                frame_index=index,
            )
            for index in range(88)
        )

        report = score_trial("all-no-op", (*background, *unresolved))

        expected = (412 * 0.10) / (412 * 0.10 + 88)
        self.assertEqual(report.score_version, AGGREGATE_SCORE_VERSION)
        self.assertAlmostEqual(report.quality_score, expected)
        self.assertLess(report.quality_score, report.pass_threshold)
        self.assertEqual(report.hit_count, 0)
        self.assertEqual(report.unresolved_count, 88)
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
