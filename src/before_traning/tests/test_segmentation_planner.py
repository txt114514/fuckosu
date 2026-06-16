from __future__ import annotations

import unittest
from pathlib import Path

from before_traning.Lib.beatmap.hit_objects import Circle
from before_traning.Lib.tools.ffmpeg import build_segment_video_args
from before_traning.Lib.video.segmentation.planner import build_segment_plans


class SegmentPlannerTests(unittest.TestCase):
    def _plans(self):
        objects = [
            Circle(3000, 3000, 100.0, 100.0),
            Circle(5000, 5000, 120.0, 100.0),
            Circle(7000, 7000, 140.0, 100.0),
            Circle(9000, 9000, 160.0, 100.0),
        ]
        return build_segment_plans(
            objects,
            approach_preempt_ratio=0.5,
            circle_size=5.0,
            min_circle_overlap_ratio=0.5,
            priority_merge_window_ms=0,
            use_priority_merge=False,
            approach_preempt_seconds=1.0,
            pre_context_jitter_seconds=0.2,
            post_context_seconds=0.4,
            video_duration_seconds=12.0,
        )

    def test_pre_context_jitter_is_stable_and_varied(self) -> None:
        first = self._plans()
        second = self._plans()

        self.assertEqual(first, second)
        contexts = [round(plan.pre_context_seconds, 6) for plan in first]
        self.assertTrue(all(0.3 <= value <= 0.7 for value in contexts))
        self.assertGreater(len(set(contexts)), 1)

    def test_segment_video_strips_audio_by_default(self) -> None:
        args = build_segment_video_args(
            Path("source.mp4"),
            Path("segment.mp4"),
            trim_start_seconds=1.0,
            trim_duration_seconds=2.0,
        )

        self.assertIn("-an", args)
        self.assertNotIn("-c:a", args)


if __name__ == "__main__":
    unittest.main()
