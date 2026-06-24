from __future__ import annotations

import unittest

from traning.lib.data import PatchMeta
from traning.lib.models import OBJECT_TYPE_NAMES
from traning.lib.training import OBJECT_TYPE_TO_ID, build_spatial_loss_targets


class SpatialTargetTests(unittest.TestCase):
    def test_circle_target_contains_center_and_approach_ring(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
