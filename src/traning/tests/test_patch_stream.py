from __future__ import annotations

import unittest

import torch

from traning.Lib.data import PatchStream


class PatchStreamTests(unittest.TestCase):
    def assert_full_coverage(self, width: int, height: int) -> None:
        stream = PatchStream(
            patch_width=512, patch_height=512, overlap_x=128, overlap_y=128
        )
        frame = torch.zeros((3, height, width))
        coverage = torch.zeros((height, width), dtype=torch.bool)
        coords = set()
        for patch, meta in stream.iter_patches(frame):
            self.assertEqual(tuple(patch.shape), (3, 512, 512))
            self.assertNotIn((meta.x0, meta.y0), coords)
            coords.add((meta.x0, meta.y0))
            coverage[meta.y0 : meta.y1, meta.x0 : meta.x1] = True
        self.assertTrue(bool(coverage.all()))

    def test_common_resolutions_are_fully_covered(self) -> None:
        self.assert_full_coverage(1484, 846)
        self.assert_full_coverage(1920, 1080)

    def test_odd_dimensions_are_fully_covered(self) -> None:
        self.assert_full_coverage(777, 515)

    def test_small_image_is_padded(self) -> None:
        stream = PatchStream(
            patch_width=512, patch_height=512, overlap_x=128, overlap_y=128
        )
        frame = torch.ones((3, 300, 255))
        patch, meta = next(stream.iter_patches(frame))
        self.assertEqual(tuple(patch.shape), (3, 512, 512))
        self.assertEqual(meta.valid_width, 255)
        self.assertEqual(meta.valid_height, 300)
        self.assertTrue(bool((patch[:, :300, :255] == 1).all()))
        self.assertTrue(bool((patch[:, 300:, :] == 0).all()))

    def test_invalid_overlap_raises(self) -> None:
        with self.assertRaises(ValueError):
            PatchStream(patch_width=512, patch_height=512, overlap_x=512, overlap_y=0)


if __name__ == "__main__":
    unittest.main()
