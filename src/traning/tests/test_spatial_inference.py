from __future__ import annotations

import unittest

import torch

from traning.conf import Settings
from traning.core.spatial import (
    SPATIAL_CPU_TASKS,
    SPATIAL_GPU_TASKS,
    run_spatial_frame_inference,
    spatial_candidate_to_dict,
)


def _tiny_settings() -> Settings:
    return Settings(
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


class SpatialInferenceTests(unittest.TestCase):
    def test_cpu_single_frame_inference_reports_cpu_gpu_split(self) -> None:
        sample = {
            "image": torch.rand(3, 96, 128),
        }
        result = run_spatial_frame_inference(
            _tiny_settings(),
            sample,
            device=torch.device("cpu"),
            max_candidates=2,
            patch_limit=1,
        )

        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.patches_processed, 1)
        self.assertEqual(result.frame_channels, 6)
        self.assertEqual(result.gpu_tasks, SPATIAL_GPU_TASKS)
        self.assertEqual(result.cpu_tasks, SPATIAL_CPU_TASKS)
        self.assertLessEqual(len(result.candidates), 2)
        self.assertIn(
            "spatial_prediction_head_per_patch",
            result.as_summary()["gpu_tasks"],
        )
        self.assertIn(
            "prediction_detach_and_canvas_fusion",
            result.as_summary()["cpu_tasks"],
        )
        if result.candidates:
            row = spatial_candidate_to_dict(result.candidates[0])
            self.assertNotIn("embedding", row)
            self.assertIn("score", row)


if __name__ == "__main__":
    unittest.main()
