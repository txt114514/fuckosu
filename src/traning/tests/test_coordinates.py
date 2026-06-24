from __future__ import annotations

import unittest

from traning.lib.data import (
    PatchMeta,
    feature_grid_to_image,
    global_to_local,
    global_to_patch_indices,
    image_to_feature_grid,
    local_to_global,
)


class CoordinateTests(unittest.TestCase):
    def test_local_global_round_trip(self) -> None:
        meta = PatchMeta(
            index=3,
            x0=384,
            y0=128,
            x1=896,
            y1=640,
            frame_width=1484,
            frame_height=846,
            valid_width=512,
            valid_height=512,
        )
        global_xy = local_to_global(meta, 20.5, 31.25)
        self.assertEqual(global_xy, (404.5, 159.25))
        self.assertEqual(global_to_local(meta, *global_xy), (20.5, 31.25))

    def test_global_to_patch_indices_returns_all_overlaps(self) -> None:
        metas = (
            PatchMeta(0, 0, 0, 512, 512, 768, 512, 512, 512),
            PatchMeta(1, 384, 0, 768, 512, 768, 512, 384, 512),
        )
        self.assertEqual(global_to_patch_indices(metas, 400, 250), (0, 1))

    def test_feature_grid_round_trip(self) -> None:
        grid = image_to_feature_grid(128.0, 64.0, stride=8)
        self.assertEqual(grid, (16.0, 8.0))
        self.assertEqual(feature_grid_to_image(*grid, stride=8), (128.0, 64.0))


if __name__ == "__main__":
    unittest.main()
