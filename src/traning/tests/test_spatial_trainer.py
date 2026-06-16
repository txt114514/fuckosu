from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from traning.conf import Settings
from traning.training import run_spatial_training


class SpatialTrainerTests(unittest.TestCase):
    def test_cpu_single_step_with_synthetic_sample(self) -> None:
        settings = Settings(
            runtime={"seed": 7, "device": "cpu"},
            input={"width": 128, "height": 96, "color_cues": "osu_basic"},
            tiling={
                "patch_width": 64,
                "patch_height": 64,
                "overlap_x": 32,
                "overlap_y": 32,
                "patch_batch_size": 1,
                "serial": True,
            },
            local_encoder={
                "stem_channels": 4,
                "feature_channels": 8,
                "output_stride": 8,
                "embedding_dim": 8,
            },
            global_encoder={
                "input_height": 64,
                "input_width": 64,
                "feature_channels": 8,
                "backbone": "lightweight_cnn",
                "pretrained": False,
                "frozen": False,
            },
            fusion={
                "mode": "gated_sparse_sampling",
                "heads": 4,
                "sampling_points": 2,
                "layers": 1,
                "hidden_dim": 16,
            },
            memory={
                "amp_dtype": "float32",
                "gradient_checkpointing": False,
                "channels_last": False,
                "allow_tf32": False,
                "cudnn_benchmark": False,
                "compile_model": False,
            },
            loader={"pin_memory": False},
        )
        sample = {
            "image": torch.rand(3, 96, 128),
            "timestamp_ms": 0.0,
            "circle_radius_osu_pixels": 20.0,
            "approach_preempt_ms": 1000.0,
            "visible_hit_objects": (
                {
                    "type": "circle",
                    "start_ms": 250,
                    "end_ms": 250,
                    "x": 128.0,
                    "y": 128.0,
                },
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = run_spatial_training(
                settings,
                device=torch.device("cpu"),
                run_dir=Path(temporary),
                max_steps=1,
                patch_limit=1,
                dataset=[sample],
            )
            self.assertEqual(result.steps, 1)
            self.assertEqual(result.last_patch_count, 1)
            self.assertTrue(math.isfinite(result.last_loss))
            self.assertTrue((Path(temporary) / "summary.txt").is_file())


if __name__ == "__main__":
    unittest.main()
