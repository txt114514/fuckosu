"""验证决策 JSONL、候选缓存与 osu!/视频坐标评分的端到端换算。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from package.coordinates import AffineOsuVideoTransform
from traning.core.optimization import (
    build_batch_gallery_request,
    score_decision_outputs,
)


class DecisionOutputScoringTests(unittest.TestCase):
    def test_scoring_rejects_missing_or_orphan_decision_frames(self) -> None:
        cache_rows = (
            {
                "sample_key": "item_0001/single_point_0001",
                "frame_index": 1,
                "timestamp_ms": 1000.0,
                "frame_width": 640,
                "frame_height": 480,
                "temporal_target": {"action": "no_op", "action_id": 0},
                "candidates": [],
            },
            {
                "sample_key": "item_0001/single_point_0001",
                "frame_index": 2,
                "timestamp_ms": 1016.0,
                "frame_width": 640,
                "frame_height": 480,
                "temporal_target": {"action": "no_op", "action_id": 0},
                "candidates": [],
            },
        )
        decisions = (
            {
                "sample_key": "item_0001/single_point_0001",
                "frame_index": 1,
                "timestamp_ms": 1000.0,
                "action": "no_op",
            },
            {
                "sample_key": "item_0001/single_point_0001",
                "frame_index": 3,
                "timestamp_ms": 1032.0,
                "action": "no_op",
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "frames.jsonl"
            decisions_path = root / "decisions.jsonl"
            cache_path.write_text(
                "".join(json.dumps(row) + "\n" for row in cache_rows),
                encoding="utf-8",
            )
            decisions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in decisions),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing_decisions=1 orphan_decisions=1",
            ):
                score_decision_outputs(
                    parameter_group_id="pg-frame-contract",
                    candidate_cache_path=cache_path,
                    decisions_path=decisions_path,
                )

    def test_scoring_uses_circle_radius_persisted_for_each_sample(self) -> None:
        transform = AffineOsuVideoTransform(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        cache_record = {
            "sample_key": "item_0001/circle_size_3",
            "frame_index": 5,
            "timestamp_ms": 1000.0,
            "frame_width": 640,
            "frame_height": 480,
            "circle_radius_osu_pixels": 40.0,
            "coordinate_transform": transform.spec(
                source="test.identity",
                status="calibrated",
            ).as_dict(),
            "temporal_target": {
                "action": "press",
                "action_id": 1,
                "target_osu_xy": [100.0, 100.0],
                "time_offset_ms": 0.0,
                "source_index": 3,
                "object_start_ms": 1000.0,
                "object_end_ms": 1000.0,
            },
            "candidates": [{"candidate_id": 0, "x": 135.0, "y": 100.0}],
        }
        decision = {
            "sample_key": "item_0001/circle_size_3",
            "frame_index": 5,
            "timestamp_ms": 1000.0,
            "action": "press",
            "action_id": 1,
            "action_probability": 0.95,
            "selected_candidate_id": 0,
            "selected_candidate_probability": 0.9,
            "time_offset_ms": 0.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "frames.jsonl"
            decisions_path = root / "decisions.jsonl"
            cache_path.write_text(json.dumps(cache_record) + "\n", encoding="utf-8")
            decisions_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")

            actual = score_decision_outputs(
                parameter_group_id="pg-real-radius",
                candidate_cache_path=cache_path,
                decisions_path=decisions_path,
            )
            legacy = score_decision_outputs(
                parameter_group_id="pg-legacy-radius",
                candidate_cache_path=cache_path,
                decisions_path=decisions_path,
                circle_radius=32.0,
            )

        self.assertEqual(actual.report.hit_count, 1)
        self.assertEqual(actual.report.unresolved_count, 0)
        self.assertEqual(
            actual.report.samples[0].metadata["circle_radius_osu_pixels"],
            40.0,
        )
        self.assertEqual(legacy.report.hit_count, 0)
        self.assertEqual(legacy.report.unresolved_count, 1)

    def test_scores_parameter_group_from_cache_and_decisions(self) -> None:
        # 决策只携带 selected_candidate_id，预测位置必须优先回查缓存中的
        # 原始视频像素，并继续传递到 gallery 请求。
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

    def test_scores_time_offset_as_frame_minus_action_boundary(self) -> None:
        # 1200 - 150 = 1050ms 落在目标有效时间附近；若 time_offset 符号
        # 反转，点击会变成 1350ms 并触发时间未命中。
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "frames.jsonl"
            decisions_path = root / "decisions.jsonl"
            cache_path.write_text(
                json.dumps(
                    {
                        "sample_key": "item_0001/slider_0001",
                        "frame_index": 42,
                        "timestamp_ms": 1200.0,
                        "frame_width": 640,
                        "frame_height": 480,
                        "temporal_target": {
                            "target_strategy": "beatmap_action_v1",
                            "action": "hold",
                            "action_id": 2,
                            "selected_candidate_id": None,
                            "target_osu_xy": [256.0, 192.0],
                            "time_offset_ms": 150.0,
                            "object_type": "slider",
                            "source_index": 8,
                            "object_start_ms": 900.0,
                            "object_end_ms": 1300.0,
                        },
                        "candidates": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            decisions_path.write_text(
                json.dumps(
                    {
                        "sample_key": "item_0001/slider_0001",
                        "frame_index": 42,
                        "timestamp_ms": 1200.0,
                        "action": "hold",
                        "action_id": 2,
                        "action_probability": 1.0,
                        "selected_candidate_id": None,
                        "selected_candidate_probability": None,
                        "predicted_xy_normalized": [0.5, 0.5],
                        "time_offset_ms": 150.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = score_decision_outputs(
                parameter_group_id="pg-offset",
                candidate_cache_path=cache_path,
                decisions_path=decisions_path,
            )

        self.assertEqual(result.report.hit_count, 1)
        self.assertEqual(result.report.unresolved_count, 0)
        self.assertGreater(result.report.quality_score, 0.9)

    def test_normalized_model_output_round_trips_through_frame_and_affine_space(
        self,
    ) -> None:
        """验证模型归一化坐标先还原训练帧像素，再经逆仿射变换回 osu。"""

        # 使用带交叉项且不在帧中心的目标，避免错误的 osu 直接归一化路径碰巧通过测试。
        transform = AffineOsuVideoTransform(((2.0, 0.1, 10.0), (-0.05, 3.0, 20.0)))
        target_osu_xy = (100.0, 50.0)
        target_video_xy = transform.osu_to_video(*target_osu_xy)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_path = root / "frames.jsonl"
            decisions_path = root / "decisions.jsonl"
            cache_path.write_text(
                json.dumps(
                    {
                        "sample_key": "item_0001/affine_0001",
                        "frame_index": 3,
                        "timestamp_ms": 1000.0,
                        "frame_width": 640,
                        "frame_height": 480,
                        "coordinate_transform": transform.spec(
                            source="test.affine",
                            status="calibrated",
                        ).as_dict(),
                        "temporal_target": {
                            "action": "press",
                            "action_id": 1,
                            "target_osu_xy": list(target_osu_xy),
                            "time_offset_ms": 0.0,
                            "source_index": 3,
                            "object_start_ms": 1000.0,
                            "object_end_ms": 1000.0,
                        },
                        "candidates": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            decisions_path.write_text(
                json.dumps(
                    {
                        "sample_key": "item_0001/affine_0001",
                        "frame_index": 3,
                        "timestamp_ms": 1000.0,
                        "action": "press",
                        "action_id": 1,
                        "predicted_xy_normalized": [
                            # 决策模型输出位于训练帧空间，不是 512×384 osu 空间。
                            target_video_xy[0] / 640.0,
                            target_video_xy[1] / 480.0,
                        ],
                        "predicted_xy_space": "model_input_normalized",
                        "time_offset_ms": 0.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = score_decision_outputs(
                parameter_group_id="pg-affine",
                candidate_cache_path=cache_path,
                decisions_path=decisions_path,
            )

        self.assertEqual(result.report.hit_count, 1)
        self.assertEqual(result.report.miss_count, 0)
        self.assertEqual(
            result.report.samples[0].metadata["predicted_video_xy"],
            target_video_xy,
        )


if __name__ == "__main__":
    unittest.main()
