from __future__ import annotations

import unittest

from traning.lib.metrics import (
    PredictedClick,
    SequenceScoreSpec,
    TargetObject,
    score_click_sequence,
)


class ClickSequenceScoringTests(unittest.TestCase):
    def test_first_passing_hit_resolves_target_once(self) -> None:
        target = TargetObject(
            target_id="circle-1",
            target_type="circle",
            start_ms=1000.0,
            end_ms=1000.0,
            x=100.0,
            y=100.0,
            source_index=1,
        )
        result = score_click_sequence(
            (target,),
            (
                PredictedClick(time_ms=850.0, x=100.0, y=100.0),
                PredictedClick(time_ms=910.0, x=100.0, y=100.0),
            ),
            circle_radius=10.0,
        )

        self.assertEqual(result.hit_count, 1)
        self.assertEqual(result.resolved_targets[0].click_index, 0)
        self.assertEqual(result.clicks[1].status, "miss")
        self.assertEqual(result.clicks[1].primary_error, "decision")
        self.assertEqual(
            result.clicks[1].error_tags,
            ("duplicate_after_hit", "better_score_after_resolution"),
        )
        self.assertEqual(result.unresolved_target_ids, ())

    def test_failed_hit_keeps_target_active_for_later_click(self) -> None:
        target = TargetObject(
            target_id="circle-1",
            target_type="circle",
            start_ms=1000.0,
            end_ms=1000.0,
            x=100.0,
            y=100.0,
        )
        result = score_click_sequence(
            (target,),
            (
                PredictedClick(time_ms=1000.0, x=120.0, y=100.0),
                PredictedClick(time_ms=1060.0, x=100.0, y=100.0),
            ),
            circle_radius=10.0,
        )

        self.assertEqual([item.status for item in result.clicks], ["miss", "hit"])
        self.assertEqual(result.clicks[0].primary_error, "spatial")
        self.assertIn("spatial_miss", result.clicks[0].error_tags)
        self.assertEqual(result.resolved_targets[0].click_index, 1)

    def test_early_click_is_attributed_to_temporal_parameters(self) -> None:
        target = TargetObject(
            target_id="circle-1",
            target_type="circle",
            start_ms=1000.0,
            end_ms=1000.0,
            x=100.0,
            y=100.0,
        )
        result = score_click_sequence(
            (target,),
            (PredictedClick(time_ms=800.0, x=100.0, y=100.0),),
            circle_radius=10.0,
        )

        self.assertEqual(result.clicks[0].status, "miss")
        self.assertEqual(result.clicks[0].primary_error, "temporal")
        self.assertEqual(result.clicks[0].error_tags, ("early_click",))
        self.assertEqual(result.clicks[0].temporal_error_ms, -200.0)

    def test_overlapping_targets_resolve_by_earliest_active_target(self) -> None:
        first = TargetObject(
            target_id="circle-1",
            target_type="circle",
            start_ms=1000.0,
            end_ms=1000.0,
            x=100.0,
            y=100.0,
            source_index=1,
        )
        second = TargetObject(
            target_id="circle-2",
            target_type="circle",
            start_ms=1030.0,
            end_ms=1030.0,
            x=100.0,
            y=100.0,
            source_index=2,
        )
        result = score_click_sequence(
            (second, first),
            (
                PredictedClick(time_ms=1000.0, x=100.0, y=100.0),
                PredictedClick(time_ms=1060.0, x=100.0, y=100.0),
            ),
            circle_radius=10.0,
        )

        self.assertEqual(
            [item.target_id for item in result.resolved_targets],
            ["circle-1", "circle-2"],
        )

    def test_click_frequency_limit_blocks_high_rate_hits(self) -> None:
        first = TargetObject(
            target_id="circle-1",
            target_type="circle",
            start_ms=1000.0,
            end_ms=1000.0,
            x=100.0,
            y=100.0,
        )
        second = TargetObject(
            target_id="circle-2",
            target_type="circle",
            start_ms=1030.0,
            end_ms=1030.0,
            x=100.0,
            y=100.0,
        )
        result = score_click_sequence(
            (first, second),
            (
                PredictedClick(time_ms=1000.0, x=100.0, y=100.0),
                PredictedClick(time_ms=1030.0, x=100.0, y=100.0),
                PredictedClick(time_ms=1060.0, x=100.0, y=100.0),
            ),
            circle_radius=10.0,
            spec=SequenceScoreSpec(min_click_interval_ms=50.0),
        )

        self.assertEqual(
            [item.status for item in result.clicks],
            ["hit", "frequency_limited", "hit"],
        )
        self.assertEqual(result.clicks[1].primary_error, "decision")
        self.assertEqual(result.clicks[1].error_tags, ("frequency_limited",))
        self.assertTrue(result.clicks[1].frequency_limited)
        self.assertEqual(result.frequency_limited_count, 1)
        self.assertEqual(result.unresolved_target_ids, ())


if __name__ == "__main__":
    unittest.main()
