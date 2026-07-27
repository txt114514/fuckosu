"""验证训练计划关键缺口的坐标、动作目标、版本和迁移回归。"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from package.coordinates import (
    AffineOsuVideoTransform,
    COORDINATE_TRANSFORM_VERSION,
    OsuVideoTransform,
    PlayfieldRect,
)
from traning.conf import Settings, load_settings
from traning.core.decision.generator import build_candidate_cache_record
from traning.core.diagnostics.oracle_ladder import _target_assignment
from traning.core.model_export import migrate_settings_file
from traning.core.temporal.trainer import _compute_temporal_loss
from traning.lib.coordinates import transform_from_settings_or_sample
from traning.state.versioning import ensure_compatible_versions


# 从训练、验证通过样本的目标中心联合拟合出的最终 osu -> 1484×846 视频仿射矩阵。
FINAL_CALIBRATED_MATRIX = (
    (2.115860914627143, 0.0011971920855575358, 242.59057485632047),
    (0.0003418231662923798, 2.1166805757239477, 16.12108357719331),
)

# 独立从原始视频 ROI 复核的 (osu 坐标, 视频像素坐标)，覆盖画面不同区域和对象类别。
OBSERVED_CONTROL_POINTS = (
    ((508.0, 237.0), (1317.5, 517.5)),
    ((80.0, 101.0), (412.0, 230.0)),
    ((395.0, 215.0), (1078.5, 471.5)),
    ((213.0, 179.0), (693.5, 395.0)),
    ((256.0, 183.0), (785.5, 404.5)),
)


class PlanGapClosureTests(unittest.TestCase):
    def test_oracle_target_assignment_uses_v2_osu_match_distance(self) -> None:
        # 构造像素最近与 osu 最近不同的 affine 诊断记录。
        # v2 的 selected id 依据 osu 距离，oracle 不得再报 selected_not_nearest。
        record = {
            "sample_key": "item/affine",
            "frame_index": 105,
            "candidates": (
                {"candidate_id": 0, "x": 10.0, "y": 10.0},
                {"candidate_id": 1, "x": 20.0, "y": 20.0},
            ),
            "temporal_target": {
                "action": "press",
                "selected_candidate_id": 1,
                "candidate_match_space": "osu",
                "candidate_match_radius_osu": 2.0,
                "candidate_match_radius_px": 8.0,
                "candidate_distances_osu": [3.0, 1.0],
                "candidate_distances_px": [1.0, 2.0],
                "candidate_match_status": "matched",
            },
        }

        assignment = _target_assignment((record,), candidate_slots=2)

        self.assertEqual(assignment["matched_candidate_frames"], 1)
        self.assertEqual(assignment["nearest_within_config_radius_frames"], 1)
        self.assertEqual(assignment["selected_not_nearest_frames"], 0)
        self.assertEqual(assignment["examples"][0]["nearest_candidate_id"], 1)
        self.assertEqual(assignment["examples"][0]["match_distance_space"], "osu")
        self.assertEqual(assignment["examples"][0]["nearest_distance_osu"], 1.0)
        self.assertEqual(assignment["examples"][0]["nearest_distance_px"], 2.0)

    def test_training_configs_use_calibrated_affine_matrix(self) -> None:
        """确保两份正式训练配置引用同一组经过校准的仿射系数。"""

        for config_path in (
            Path("configs/model_full_small_vram.yaml"),
            Path("configs/model_small_vram.yaml"),
        ):
            with self.subTest(config=str(config_path)):
                settings = load_settings(config_path)
                self.assertEqual(settings.coordinate_transform.mode, "affine_matrix")
                self.assertIsNotNone(settings.coordinate_transform.matrix)
                transform, spec = transform_from_settings_or_sample(
                    settings,
                    frame_width=1484,
                    frame_height=846,
                )
                self.assertEqual(spec.source, "settings.affine_matrix")
                self.assertEqual(spec.transform_status, "calibrated")
                self.assertIsNotNone(spec.matrix)
                self.assertEqual(spec.matrix, FINAL_CALIBRATED_MATRIX)
                point = transform.osu_to_video(425.0, 98.0)
                restored = transform.video_to_osu(*point)
                self.assertAlmostEqual(restored[0], 425.0, places=6)
                self.assertAlmostEqual(restored[1], 98.0, places=6)

    def test_final_affine_matches_independent_observed_control_points(self) -> None:
        """用独立观测点约束拟合结果，防止只对某一张示例图片过拟合。"""

        transform = AffineOsuVideoTransform(FINAL_CALIBRATED_MATRIX)

        for osu_xy, observed_video_xy in OBSERVED_CONTROL_POINTS:
            with self.subTest(osu_xy=osu_xy, observed_video_xy=observed_video_xy):
                predicted_video_xy = transform.osu_to_video(*osu_xy)
                self.assertLessEqual(
                    # 4 px 容差覆盖圆心检测的像素取整和边缘拟合误差。
                    math.dist(predicted_video_xy, observed_video_xy),
                    4.0,
                    msg=(
                        f"osu={osu_xy} predicted={predicted_video_xy} "
                        f"observed={observed_video_xy}"
                    ),
                )

    def test_explicit_non_centered_playfield_round_trip(self) -> None:
        transform = OsuVideoTransform.from_rect(
            PlayfieldRect(left=111, top=27, width=1024, height=768)
        )
        for point in ((0.0, 0.0), (512.0, 384.0), (128.5, 240.25)):
            video = transform.osu_to_video(*point)
            restored = transform.video_to_osu(*video)
            self.assertAlmostEqual(restored[0], point[0], places=6)
            self.assertAlmostEqual(restored[1], point[1], places=6)

    def test_affine_playfield_round_trip(self) -> None:
        transform = AffineOsuVideoTransform.from_rows(
            ((2.25, 0.05, 160.0), (-0.03, 2.2, 7.0))
        )
        for point in ((0.0, 0.0), (512.0, 384.0), (128.5, 240.25)):
            video = transform.osu_to_video(*point)
            restored = transform.video_to_osu(*video)
            self.assertAlmostEqual(restored[0], point[0], places=6)
            self.assertAlmostEqual(restored[1], point[1], places=6)

    def test_source_rect_applies_crop_offset_before_video_mapping(self) -> None:
        settings = Settings(
            coordinate_transform={
                "mode": "explicit_source_rect",
                "playfield_rect": {
                    "left": 352,
                    "top": 167,
                    "width": 1128,
                    "height": 846,
                },
                "crop_rect": {
                    "left": 174,
                    "top": 167,
                    "width": 1484,
                    "height": 846,
                },
            }
        )
        transform, spec = transform_from_settings_or_sample(
            settings,
            frame_width=1484,
            frame_height=846,
        )
        self.assertEqual(spec.source, "settings.explicit_source_rect")
        self.assertEqual(transform.osu_to_video(0, 0), (178.0, 0.0))
        self.assertEqual(transform.osu_to_video(512, 384), (1306.0, 846.0))

    def test_action_targets_include_circle_release_slider_repeat_and_spinner(
        self,
    ) -> None:
        settings = Settings(
            coordinate_transform={
                "mode": "explicit_rect",
                "playfield_rect": {"left": 10, "top": 20, "width": 1024, "height": 768},
            }
        )
        candidates = ()
        base = {
            "sample_key": "item/segment",
            "frame_index": 1,
            "image": torch.zeros(3, 100, 120),
            "coordinate_transform": {
                "version": COORDINATE_TRANSFORM_VERSION,
                "rect": {"left": 10, "top": 20, "width": 1024, "height": 768},
            },
        }
        # 三个时间点分别命中 circle 松开窗口、slider repeat 反向端点和
        # spinner 持续区间，覆盖不同对象共享 action 编码时的边界分支。
        circle_release = build_candidate_cache_record(
            base
            | {
                "timestamp_ms": 106.0,
                "hit_objects": (
                    {
                        "type": "circle",
                        "start_ms": 100,
                        "end_ms": 100,
                        "x": 256,
                        "y": 192,
                        "source_index": 7,
                    },
                ),
            },
            candidates,
            (),
            frame_width=120,
            frame_height=100,
            device="cpu",
            patches_processed=1,
            frame_channels=3,
            save_dtype="float32",
            low_confidence_threshold=0.6,
            close_score_margin=0.05,
            slider_attach_distance_px=48,
            action_window_ms=5,
            settings=settings,
        )["temporal_target"]
        self.assertEqual(circle_release["action"], "release")

        slider_repeat = build_candidate_cache_record(
            base
            | {
                "timestamp_ms": 150.0,
                "hit_objects": (
                    {
                        "type": "slider",
                        "start_ms": 100,
                        "end_ms": 200,
                        "x": 10,
                        "y": 10,
                        "path": ((10, 10), (100, 10)),
                        "repeats": 2,
                        "source_index": 8,
                    },
                ),
            },
            candidates,
            (),
            frame_width=120,
            frame_height=100,
            device="cpu",
            patches_processed=1,
            frame_channels=3,
            save_dtype="float32",
            low_confidence_threshold=0.6,
            close_score_margin=0.05,
            slider_attach_distance_px=48,
            action_window_ms=10,
            settings=settings,
        )["temporal_target"]
        self.assertEqual(slider_repeat["action"], "press")
        self.assertEqual(slider_repeat["target_osu_xy"], [100.0, 10.0])

        spinner_hold = build_candidate_cache_record(
            base
            | {
                "timestamp_ms": 150.0,
                "hit_objects": (
                    {
                        "type": "spinner",
                        "start_ms": 100,
                        "end_ms": 200,
                        "source_index": 9,
                    },
                ),
            },
            candidates,
            (),
            frame_width=120,
            frame_height=100,
            device="cpu",
            patches_processed=1,
            frame_channels=3,
            save_dtype="float32",
            low_confidence_threshold=0.6,
            close_score_margin=0.05,
            slider_attach_distance_px=48,
            action_window_ms=10,
            settings=settings,
        )["temporal_target"]
        self.assertEqual(spinner_hold["action"], "hold")

    def test_temporal_loss_weights_change_combined_loss(self) -> None:
        class Weights:
            action = 1.0
            candidate = 1.0
            xy = 1.0
            time_offset = 0.01

        class TimeHeavy:
            action = 1.0
            candidate = 1.0
            xy = 1.0
            time_offset = 1.0

        # 固定同一组 logits/回归误差，只改变 time_offset 权重，避免模型
        # 随机性或其他 loss 分量掩盖配置权重是否真正参与总损失。
        output = type(
            "Output",
            (),
            {
                "action_logits": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
                "selected_candidate_logits": torch.tensor([[0.0, 1.0]]),
                "x": torch.tensor([[0.0]]),
                "y": torch.tensor([[0.0]]),
                "time_offset_ms": torch.tensor([[0.0]]),
            },
        )()
        args = {
            "action_target": torch.tensor([1]),
            "selected_candidate_target": torch.tensor([1]),
            "xy_target": torch.tensor([[1.0, 1.0]]),
            "time_offset_target": torch.tensor([[10.0]]),
            "frame_mask": torch.tensor([True]),
        }
        base, _ = _compute_temporal_loss([output], weights=Weights, **args)
        changed, _ = _compute_temporal_loss([output], weights=TimeHeavy, **args)
        self.assertGreater(float(changed), float(base))

    def test_version_mismatch_blocks_without_override(self) -> None:
        ok, mismatches = ensure_compatible_versions(
            {"dataset_version": "a", "score_version": "s1"},
            {"dataset_version": "b", "score_version": "s1"},
        )
        self.assertFalse(ok)
        self.assertEqual(mismatches, ("dataset_version",))
        ok, mismatches = ensure_compatible_versions(
            {"dataset_version": "a"},
            {"dataset_version": "b"},
            override=True,
        )
        self.assertTrue(ok)
        self.assertEqual(mismatches, ("dataset_version",))

    def test_transform_fingerprint_mismatch_blocks_without_override(self) -> None:
        """协议版本相同但方程摘要不同，也必须阻止缓存或检查点复用。"""

        left = {
            "transform_version": COORDINATE_TRANSFORM_VERSION,
            "transform_fingerprint": "affine-sha256-a",
        }
        right = {
            "transform_version": COORDINATE_TRANSFORM_VERSION,
            "transform_fingerprint": "affine-sha256-b",
        }

        ok, mismatches = ensure_compatible_versions(left, right)
        self.assertFalse(ok)
        self.assertEqual(mismatches, ("transform_fingerprint",))

        ok, mismatches = ensure_compatible_versions(left, right, override=True)
        self.assertTrue(ok)
        self.assertEqual(mismatches, ("transform_fingerprint",))

    def test_settings_migration_adds_schema_and_transform(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Path(tmpdir) / "settings.yaml"
            settings.write_text("runtime:\n  device: cpu\n", encoding="utf-8")
            migrated, log = migrate_settings_file(settings)
            data = json.dumps(log)
            self.assertTrue(migrated.exists())
            self.assertIn("add_legacy_transform", data)


if __name__ == "__main__":
    unittest.main()
