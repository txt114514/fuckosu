"""验证 circle、slider、spinner 的 patch 级空间监督光栅化。"""

from __future__ import annotations

import unittest

from package.coordinates import AffineOsuVideoTransform
from traning.lib.data import PatchMeta
from traning.lib.models import OBJECT_TYPE_NAMES
from traning.lib.training import OBJECT_TYPE_TO_ID, build_spatial_loss_targets


class SpatialTargetTests(unittest.TestCase):
    def test_circle_target_contains_center_and_approach_ring(self) -> None:
        # 目标位于非零 patch 原点内，时间处于 approach 阶段，因而一个样本
        # 必须同时产生中心监督和 approach ring 监督。
        meta = PatchMeta(
            0,
            192,
            128,
            320,
            256,
            512,
            384,
            128,
            128,
            128,
            128,
        )
        sample = {
            "timestamp_ms": 500.0,
            "circle_radius_osu_pixels": 20.0,
            "approach_preempt_ms": 1000.0,
            "visible_hit_objects": (
                {
                    "type": "circle",
                    "start_ms": 1000,
                    "end_ms": 1000,
                    "x": 256.0,
                    "y": 192.0,
                },
            ),
        }
        target = build_spatial_loss_targets(sample, meta, (16, 16))
        self.assertGreater(float(target.center_heatmap.max()), 0.9)
        self.assertGreater(int(target.ring_mask.sum()), 0)
        self.assertGreater(float(target.ring_radius.max()), 0.0)
        types = set(target.object_type.flatten().tolist())
        self.assertIn(OBJECT_TYPE_TO_ID["hit_circle"], types)
        self.assertIn(OBJECT_TYPE_TO_ID["approach_circle"], types)

    def test_slider_target_contains_body_direction_head_and_tail(self) -> None:
        meta = PatchMeta(0, 0, 0, 512, 384, 512, 384, 512, 384, 512, 384)
        # 水平直线让方向期望可解析为 (±1, 0)，便于捕获通道次序或坐标轴颠倒。
        sample = {
            "timestamp_ms": 0.0,
            "circle_radius_osu_pixels": 20.0,
            "visible_hit_objects": (
                {
                    "type": "slider",
                    "start_ms": 100,
                    "end_ms": 400,
                    "path": ((64.0, 192.0), (448.0, 192.0)),
                    "repeats": 1,
                },
            ),
        }
        target = build_spatial_loss_targets(sample, meta, (48, 64))
        self.assertGreater(int(target.slider_mask.sum()), 0)
        mask = target.slider_mask[0, 0].bool()
        horizontal = target.slider_direction[0, 0][mask]
        vertical = target.slider_direction[0, 1][mask]
        self.assertGreater(float(horizontal.abs().mean()), 0.9)
        self.assertLess(float(vertical.abs().mean()), 0.1)
        types = set(target.object_type.flatten().tolist())
        self.assertIn(OBJECT_TYPE_TO_ID["slider_head"], types)
        self.assertIn(OBJECT_TYPE_TO_ID["slider_body"], types)
        self.assertIn(OBJECT_TYPE_TO_ID["slider_tail"], types)

    def test_spinner_target_marks_valid_patch_area(self) -> None:
        # patch 张量含右侧和底部 padding，spinner dense mask 不得监督补齐区域。
        meta = PatchMeta(0, 0, 0, 96, 80, 128, 96, 96, 80, 128, 96)
        sample = {
            "timestamp_ms": 0.0,
            "visible_hit_objects": (
                {"type": "spinner", "start_ms": 0, "end_ms": 1000},
            ),
        }
        target = build_spatial_loss_targets(sample, meta, (12, 16))
        self.assertGreater(int(target.spinner_mask.sum()), 0)
        self.assertLess(int(target.spinner_mask.sum()), 12 * 16)
        spinner_id = OBJECT_TYPE_NAMES.index("spinner")
        self.assertIn(spinner_id, set(target.object_type.flatten().tolist()))

    def test_affine_spinner_target_uses_transformed_playfield_center(self) -> None:
        """验证 spinner 的 dense target 使用仿射后的 osu 中心，而非轴对齐字段。"""

        matrix = (
            (2.115860914627143, 0.0011971920855575358, 242.59057485632047),
            (0.0003418231662923798, 2.1166805757239477, 16.12108357719331),
        )
        transform = AffineOsuVideoTransform(matrix)
        frame_width = 1484
        frame_height = 846
        feature_width = 148
        feature_height = 84
        meta = PatchMeta(
            0,
            0,
            0,
            frame_width,
            frame_height,
            frame_width,
            frame_height,
            frame_width,
            frame_height,
        )
        spinner = {"type": "spinner", "start_ms": 0, "end_ms": 1000}
        sample = {
            "timestamp_ms": 0.0,
            "visible_hit_objects": (spinner,),
            "coordinate_transform": transform.spec(
                source="test.affine",
                status="calibrated",
            ).as_dict(),
        }

        target = build_spatial_loss_targets(
            sample,
            meta,
            (feature_height, feature_width),
        )

        flat_index = int(target.center_heatmap.argmax().item())
        row, column = divmod(flat_index, feature_width)
        peak_xy = (
            (column + 0.5) * frame_width / feature_width,
            (row + 0.5) * frame_height / feature_height,
        )
        expected_xy = transform.osu_to_video(256.0, 192.0)
        # heatmap 单元代表一个像素区域，因此允许一个特征网格宽高以内的量化误差。
        self.assertAlmostEqual(
            peak_xy[0],
            expected_xy[0],
            delta=frame_width / feature_width,
        )
        self.assertAlmostEqual(
            peak_xy[1],
            expected_xy[1],
            delta=frame_height / feature_height,
        )


if __name__ == "__main__":
    unittest.main()
