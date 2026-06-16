from __future__ import annotations

import unittest

import torch

from traning.models import OBJECT_TYPE_NAMES, SpatialPredictionHead


class SpatialModelTests(unittest.TestCase):
    def test_prediction_head_outputs_all_required_tasks(self) -> None:
        head = SpatialPredictionHead(16, embedding_dim=12)
        output = head(torch.randn(2, 16, 8, 8))
        self.assertEqual(output.center_heatmap.shape, (2, 1, 8, 8))
        self.assertEqual(output.visible_heatmap.shape, (2, 1, 8, 8))
        self.assertEqual(output.xy_offset.shape, (2, 2, 8, 8))
        self.assertEqual(
            output.object_type_logits.shape, (2, len(OBJECT_TYPE_NAMES), 8, 8)
        )
        self.assertEqual(output.ring_mask.shape, (2, 1, 8, 8))
        self.assertEqual(output.slider_direction.shape, (2, 2, 8, 8))
        self.assertEqual(output.spinner_mask.shape, (2, 1, 8, 8))
        self.assertEqual(output.candidate_embedding.shape, (2, 12, 8, 8))


if __name__ == "__main__":
    unittest.main()
