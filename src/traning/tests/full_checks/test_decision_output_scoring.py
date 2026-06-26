from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from traning.core.optimization import build_batch_gallery_request, score_decision_outputs


class DecisionOutputScoringTests(unittest.TestCase):
    def test_scores_parameter_group_from_cache_and_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "frames.jsonl"
            decisions_path = root / "decisions.jsonl"
            cache_path.write_text(
                json.dumps(
                    {
                        "sample_key": "item_0001/single_point_0001",
                        "frame_index": 7,
                        "timestamp_ms": 1000.0,
                        "frame_width": 640,
                        "frame_height": 480,
                        "temporal_target": {
                            "target_strategy": "beatmap_action_v1",
                            "action": "press",
                            "action_id": 1,
                            "selected_candidate_id": 0,
                            "target_osu_xy": [256.0, 192.0],
                            "time_offset_ms": 0.0,
                            "source_index": 3,
                            "object_start_ms": 1000.0,
                            "object_end_ms": 1000.0,
                        },
                        "candidates": [
                            {
                                "candidate_id": 0,
                                "x": 320.0,
                                "y": 240.0,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            decisions_path.write_text(
                json.dumps(
                    {
                        "sample_key": "item_0001/single_point_0001",
                        "frame_index": 7,
                        "timestamp_ms": 1000.0,
                        "action": "press",
                        "action_id": 1,
                        "action_probability": 0.99,
                        "selected_candidate_id": 0,
                        "selected_candidate_probability": 0.95,
                        "time_offset_ms": 0.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = score_decision_outputs(
                parameter_group_id="pg-0007",
                candidate_cache_path=cache_path,
                decisions_path=decisions_path,
            )

        self.assertEqual(result.parameter_group_id, "pg-0007")
        self.assertEqual(result.report.target_count, 1)
        self.assertEqual(result.report.hit_count, 1)
        self.assertEqual(result.report.miss_count, 0)
        self.assertEqual(
            result.report.samples[0].metadata["predicted_video_xy"],
            (320.0, 240.0),
        )
        self.assertGreater(result.report.quality_score, 0.9)
        self.assertEqual(result.as_summary()["action_frames"], 1)
        gallery_request = build_batch_gallery_request(result.report)
        self.assertEqual(
            gallery_request.best_trial.frames[0].predicted_video_xy,
            (320.0, 240.0),
        )


if __name__ == "__main__":
    unittest.main()
