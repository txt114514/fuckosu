from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from traning.data import PatchMeta
from traning.models import OBJECT_TYPE_NAMES, SpatialPrediction
from traning.training import SpatialPredictionCanvas, decode_spatial_candidates


def _prediction(
    *,
    height: int = 16,
    width: int = 16,
    embedding_dim: int = 4,
) -> SpatialPrediction:
    object_types = len(OBJECT_TYPE_NAMES)
    return SpatialPrediction(
        center_heatmap=torch.full((1, 1, height, width), -12.0),
        visible_heatmap=torch.full((1, 1, height, width), 8.0),
        xy_offset=torch.zeros((1, 2, height, width)),
        object_type_logits=torch.full((1, object_types, height, width), -8.0),
        ring_mask=torch.full((1, 1, height, width), -12.0),
        ring_radius=torch.zeros((1, 1, height, width)),
        slider_mask=torch.full((1, 1, height, width), -12.0),
        slider_direction=F.normalize(torch.ones((1, 2, height, width)), dim=1),
        spinner_mask=torch.full((1, 1, height, width), -12.0),
        candidate_embedding=F.normalize(
            torch.ones((1, embedding_dim, height, width)),
            dim=1,
        ),
    )


class SpatialDecodeTests(unittest.TestCase):
    def test_canvas_decodes_global_candidate_with_offset_and_type(self) -> None:
        prediction = _prediction()
        row, col = 5, 4
        prediction.center_heatmap[0, 0, row, col] = 12.0
        prediction.xy_offset[0, 0, row, col] = 0.25
        prediction.xy_offset[0, 1, row, col] = -0.25
        prediction.object_type_logits[0, OBJECT_TYPE_NAMES.index("hit_circle"), row, col] = 10.0
        prediction.ring_mask[0, 0, row, col] = 8.0
        prediction.ring_radius[0, 0, row, col] = 3.0

        canvas = SpatialPredictionCanvas(
            frame_width=256,
            frame_height=192,
            stride=8,
            embedding_dim=4,
            feather_edges=False,
        )
        meta = PatchMeta(0, 64, 32, 192, 160, 256, 192, 128, 128, 128, 128)
        canvas.write_patch(prediction, meta)
        candidates = decode_spatial_candidates(
            canvas.to_maps(),
            max_candidates=4,
            score_threshold=0.5,
            nms_radius_px=16.0,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.object_type, "hit_circle")
        self.assertAlmostEqual(candidate.x, 64 + (col + 0.75) * 8, places=4)
        self.assertAlmostEqual(candidate.y, 32 + (row + 0.25) * 8, places=4)
        self.assertGreater(candidate.score, 0.9)
        self.assertAlmostEqual(candidate.ring_radius_px, 24.0, places=4)

    def test_padding_region_is_not_written_to_global_canvas(self) -> None:
        prediction = _prediction(height=8, width=8)
        hit_circle = OBJECT_TYPE_NAMES.index("hit_circle")
        prediction.center_heatmap[0, 0, 2, 1] = 12.0
        prediction.object_type_logits[0, hit_circle, 2, 1] = 10.0
        prediction.center_heatmap[0, 0, 2, 6] = 12.0
        prediction.object_type_logits[0, hit_circle, 2, 6] = 10.0

        canvas = SpatialPredictionCanvas(
            frame_width=40,
            frame_height=64,
            stride=8,
            embedding_dim=4,
            feather_edges=False,
        )
        meta = PatchMeta(0, 0, 0, 24, 64, 40, 64, 24, 64, 64, 64)
        canvas.write_patch(prediction, meta)
        candidates = decode_spatial_candidates(
            canvas.to_maps(),
            max_candidates=8,
            score_threshold=0.5,
            nms_radius_px=0.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertLess(candidates[0].x, 24.0)

    def test_decode_applies_nms(self) -> None:
        prediction = _prediction()
        hit_circle = OBJECT_TYPE_NAMES.index("hit_circle")
        for row, col in ((4, 4), (5, 5)):
            prediction.center_heatmap[0, 0, row, col] = 12.0
            prediction.object_type_logits[0, hit_circle, row, col] = 10.0

        canvas = SpatialPredictionCanvas(
            frame_width=128,
            frame_height=128,
            stride=8,
            embedding_dim=4,
            feather_edges=False,
        )
        canvas.write_patch(
            prediction,
            PatchMeta(0, 0, 0, 128, 128, 128, 128, 128, 128, 128, 128),
        )
        candidates = decode_spatial_candidates(
            canvas.to_maps(),
            max_candidates=8,
            score_threshold=0.5,
            nms_radius_px=20.0,
        )
        self.assertEqual(len(candidates), 1)


if __name__ == "__main__":
    unittest.main()
