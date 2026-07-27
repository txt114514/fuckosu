"""验证图集请求模型的字段校验、版本一致性和最佳 trial 选择。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from traning.state import BatchGalleryRequest, FrameEvaluation


class BatchGalleryRequestTests(unittest.TestCase):
    def test_frame_evaluation_accepts_error_attribution(self) -> None:
        frame = FrameEvaluation.model_validate(
            {
                "sample_key": "item_000001/segment_000001",
                "frame_index": 12,
                "passed": False,
                "primary_error": "decision",
                "error_tags": [
                    "duplicate_after_hit",
                    "better_score_after_resolution",
                ],
                "spatial_error": 0.0,
                "temporal_error_ms": -90.0,
                "frequency_limited": False,
            }
        )

        self.assertEqual(frame.primary_error, "decision")
        self.assertEqual(
            frame.error_tags,
            ("duplicate_after_hit", "better_score_after_resolution"),
        )

    def test_trials_must_share_score_version(self) -> None:
        # 分数看似都合法但来自不同协议，必须在选择 best trial 之前拒绝，
        # 否则数值大小没有可比性。
        with self.assertRaises(ValidationError):
            BatchGalleryRequest.model_validate(
                {
                    "batch_id": "mixed",
                    "trials": [
                        {
                            "trial_id": "a",
                            "score": 0.9,
                            "score_version": "point-slider-v2+macro-v1",
                            "parameters": {},
                        },
                        {
                            "trial_id": "b",
                            "score": 0.8,
                            "score_version": "external",
                            "parameters": {},
                        },
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
