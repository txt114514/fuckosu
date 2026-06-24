from __future__ import annotations

import unittest

from traning.lib.metrics import (
    ScoreSpec,
    score_point,
    score_slider,
    spatial_coefficient,
    temporal_coefficient,
)


class PointScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ScoreSpec()

    def test_spatial_bonus_clamps_inside_sixty_percent(self) -> None:
        self.assertEqual(
            spatial_coefficient(0.0),
            spatial_coefficient(0.6),
        )
        self.assertGreater(spatial_coefficient(0.6), 1.0)
        self.assertEqual(spatial_coefficient(1.0), 1.0)

    def test_spatial_comfort_and_zero_boundaries(self) -> None:
        self.assertGreater(spatial_coefficient(1.01), 0.0)
        self.assertLessEqual(
            spatial_coefficient(1.01),
            self.spec.comfort_score_max,
        )
        self.assertEqual(spatial_coefficient(1.5), 0.0)

    def test_temporal_boundaries_follow_v2_bands(self) -> None:
        values = [
            temporal_coefficient(error)
            for error in (20, 50, 100, 150, 150.1, 200)
        ]
        self.assertEqual(values[0], self.spec.maximum_coefficient)
        self.assertEqual(values[1], 1.0)
        self.assertEqual(values[2], self.spec.temporal_excellent_score)
        self.assertEqual(values[3], self.spec.temporal_pass_score)
        self.assertLess(values[4], self.spec.comfort_score_max)
        self.assertGreater(values[4], 0.0)
        self.assertEqual(values[5], 0.0)

    def test_point_pass_requires_space_and_time(self) -> None:
        passing = score_point(
            (0.0, 0.0),
            (10.0, 0.0),
            circle_radius=10.0,
            reference_time_ms=1000.0,
            predicted_time_ms=1150.0,
        )
        spatial_failure = score_point(
            (0.0, 0.0),
            (10.1, 0.0),
            circle_radius=10.0,
            reference_time_ms=1000.0,
            predicted_time_ms=1000.0,
        )
        temporal_failure = score_point(
            (0.0, 0.0),
            (0.0, 0.0),
            circle_radius=10.0,
            reference_time_ms=1000.0,
            predicted_time_ms=1150.1,
        )
        self.assertTrue(passing.passed)
        self.assertFalse(spatial_failure.passed)
        self.assertFalse(temporal_failure.passed)
        self.assertAlmostEqual(
            passing.score.raw,
            (
                passing.score.spatial
                + passing.score.temporal
                + passing.score.spatial * passing.score.temporal
            ),
        )


class SliderScoringTests(unittest.TestCase):
    def test_slider_uses_first_path_point_as_missing_head(self) -> None:
        path = ((5.0, 7.0), (20.0, 7.0))
        result = score_slider(
            None,
            None,
            path,
            path,
            circle_radius=5.0,
            reference_start_ms=1000.0,
            predicted_start_ms=1000.0,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.head.distance, 0.0)

    def test_slider_requires_bidirectional_path_match(self) -> None:
        reference = ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))
        passing = score_slider(
            (0.0, 0.0),
            (0.0, 0.0),
            reference,
            reference,
            circle_radius=5.0,
            reference_start_ms=1000.0,
            predicted_start_ms=1000.0,
        )
        stray_prediction = score_slider(
            (0.0, 0.0),
            (0.0, 0.0),
            reference,
            (*reference, (100.0, 100.0)),
            circle_radius=5.0,
            reference_start_ms=1000.0,
            predicted_start_ms=1000.0,
        )
        missing_path = score_slider(
            (0.0, 0.0),
            (0.0, 0.0),
            reference,
            ((0.0, 0.0),),
            circle_radius=5.0,
            reference_start_ms=1000.0,
            predicted_start_ms=1000.0,
        )
        self.assertTrue(passing.passed)
        self.assertEqual(passing.path.dilation_radius, 7.5)
        self.assertEqual(passing.path.reference_coverage, 1.0)
        self.assertEqual(passing.path.prediction_precision, 1.0)
        self.assertFalse(stray_prediction.passed)
        self.assertFalse(missing_path.passed)
        self.assertLessEqual(
            stray_prediction.score.spatial,
            ScoreSpec().comfort_score_max,
        )
        self.assertLessEqual(
            missing_path.score.spatial,
            ScoreSpec().comfort_score_max,
        )

    def test_slider_corridor_uses_one_point_five_radius(self) -> None:
        reference = ((0.0, 0.0), (100.0, 0.0))
        inside_corridor = score_slider(
            (0.0, 0.0),
            (0.0, 0.0),
            reference,
            ((0.0, 14.9), (100.0, 14.9)),
            circle_radius=10.0,
            reference_start_ms=0.0,
            predicted_start_ms=0.0,
        )
        outside_corridor = score_slider(
            (0.0, 0.0),
            (0.0, 0.0),
            reference,
            ((0.0, 15.1), (100.0, 15.1)),
            circle_radius=10.0,
            reference_start_ms=0.0,
            predicted_start_ms=0.0,
        )
        self.assertTrue(inside_corridor.path.passed)
        self.assertFalse(outside_corridor.path.passed)


if __name__ == "__main__":
    unittest.main()
