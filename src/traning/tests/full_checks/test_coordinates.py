"""验证 patch、特征网格、视频帧和 osu! 坐标之间的换算。"""

from __future__ import annotations

import unittest

from package.coordinates import (
    CoordinateTransformChain,
    ImageSize,
    OSU_PLAYFIELD_HEIGHT,
    OSU_PLAYFIELD_WIDTH,
    PlayfieldRect,
    ScreenTransform,
)
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

    def test_beatmap_video_input_round_trip_boundaries_and_center(self) -> None:
        chain = CoordinateTransformChain(
            source_size=ImageSize(width=1484, height=846),
            crop_rect=PlayfieldRect(left=100, top=50, width=1200, height=700),
            resized_size=ImageSize(width=600, height=350),
            playfield_source_rect=PlayfieldRect(left=220, top=110, width=1024, height=768),
            source="test_geometry",
        )
        # 四角覆盖方向、缩放与偏移，中心点额外捕获错误的半像素或中心对齐。
        points = (
            (0.0, 0.0),
            (OSU_PLAYFIELD_WIDTH, 0.0),
            (0.0, OSU_PLAYFIELD_HEIGHT),
            (OSU_PLAYFIELD_WIDTH, OSU_PLAYFIELD_HEIGHT),
            (OSU_PLAYFIELD_WIDTH / 2.0, OSU_PLAYFIELD_HEIGHT / 2.0),
        )

        for beatmap_x, beatmap_y in points:
            with self.subTest(point=(beatmap_x, beatmap_y)):
                source = chain.osu_to_source(beatmap_x, beatmap_y)
                self.assertAlmostEqual(chain.source_to_osu(*source)[0], beatmap_x)
                self.assertAlmostEqual(chain.source_to_osu(*source)[1], beatmap_y)
                model_input = chain.osu_to_model_input(beatmap_x, beatmap_y)
                restored = chain.model_input_to_osu(*model_input)
                self.assertAlmostEqual(restored[0], beatmap_x)
                self.assertAlmostEqual(restored[1], beatmap_y)

    def test_coordinate_chain_random_property_round_trip(self) -> None:
        chain = CoordinateTransformChain(
            source_size=ImageSize(width=1920, height=1080),
            crop_rect=PlayfieldRect(left=80, top=40, width=1600, height=900),
            resized_size=ImageSize(width=800, height=450),
            playfield_source_rect=PlayfieldRect(left=180, top=120, width=1280, height=960),
            source="property_test",
        )

        # 互质步长生成可复现且分散的控制点，不依赖随机模块状态，同时
        # 覆盖非整数坐标在 crop/resize 往返中的精度。
        max_error = 0.0
        for index in range(128):
            beatmap_x = (index * 37 % 512) + 0.125
            beatmap_y = (index * 53 % 384) + 0.25
            restored = chain.training_frame_to_osu(
                *chain.osu_to_training_frame(beatmap_x, beatmap_y)
            )
            max_error = max(
                max_error,
                abs(restored[0] - beatmap_x),
                abs(restored[1] - beatmap_y),
            )

        self.assertLess(max_error, 1e-9)

    def test_screen_mapping_uses_beatmap_as_authority(self) -> None:
        screen = ScreenTransform.from_rect(
            PlayfieldRect(left=320, top=180, width=1024, height=768)
        )

        self.assertEqual(screen.osu_to_screen(0.0, 0.0), (320.0, 180.0))
        self.assertEqual(
            screen.osu_to_screen(OSU_PLAYFIELD_WIDTH, OSU_PLAYFIELD_HEIGHT),
            (1344.0, 948.0),
        )
        self.assertEqual(
            screen.osu_to_screen(OSU_PLAYFIELD_WIDTH / 2.0, OSU_PLAYFIELD_HEIGHT / 2.0),
            (832.0, 564.0),
        )
        self.assertEqual(screen.screen_to_osu(832.0, 564.0), (256.0, 192.0))


if __name__ == "__main__":
    unittest.main()
