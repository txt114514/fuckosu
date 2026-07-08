from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from package.coordinates import COORDINATE_TRANSFORM_VERSION, OsuVideoTransform, PlayfieldRect
from traning.conf import Settings, load_settings
from traning.core.decision.generator import build_candidate_cache_record
from traning.core.model_export import migrate_settings_file
from traning.core.temporal.trainer import _compute_temporal_loss
from traning.lib.coordinates import transform_from_settings_or_sample
from traning.state.versioning import ensure_compatible_versions


class PlanGapClosureTests(unittest.TestCase):
    def test_training_configs_use_explicit_cropped_playfield_rect(self) -> None:
        for config_path in (
            Path("configs/model_full_small_vram.yaml"),
            Path("configs/model_small_vram.yaml"),
        ):
            with self.subTest(config=str(config_path)):
                settings = load_settings(config_path)
                self.assertEqual(settings.coordinate_transform.mode, "explicit_source_rect")
                rect = settings.coordinate_transform.playfield_rect
                crop = settings.coordinate_transform.crop_rect
                self.assertIsNotNone(rect)
                self.assertIsNotNone(crop)
                transform = OsuVideoTransform.from_rect(
                    PlayfieldRect(
                        left=rect.left - crop.left,
                        top=rect.top - crop.top,
                        width=rect.width,
                        height=rect.height,
                    )
                )
                self.assertEqual(transform.osu_to_video(0, 0), (178.0, 0.0))
                self.assertEqual(transform.osu_to_video(512, 384), (1306.0, 846.0))

    def test_explicit_non_centered_playfield_round_trip(self) -> None:
        transform = OsuVideoTransform.from_rect(
            PlayfieldRect(left=111, top=27, width=1024, height=768)
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

    def test_action_targets_include_circle_release_slider_repeat_and_spinner(self) -> None:
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
        circle_release = build_candidate_cache_record(
            base
            | {
                "timestamp_ms": 106.0,
                "hit_objects": ({"type": "circle", "start_ms": 100, "end_ms": 100, "x": 256, "y": 192, "source_index": 7},),
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
                "hit_objects": ({"type": "slider", "start_ms": 100, "end_ms": 200, "x": 10, "y": 10, "path": ((10, 10), (100, 10)), "repeats": 2, "source_index": 8},),
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
                "hit_objects": ({"type": "spinner", "start_ms": 100, "end_ms": 200, "source_index": 9},),
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
