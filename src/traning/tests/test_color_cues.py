from __future__ import annotations

import unittest

import torch

from traning.conf import Settings
from traning.Lib.data import (
    append_color_cues,
    color_cue_channel_count,
    extract_osu_basic_color_cues,
)
from traning.Lib.models import build_model_stack


class ColorCueTests(unittest.TestCase):
    def test_osu_basic_cues_highlight_colored_target_and_white_number(self) -> None:
        frame = torch.zeros(3, 32, 32)
        frame[:, :, :] = torch.tensor([0.05, 0.06, 0.08]).view(3, 1, 1)
        frame[:, 8:24, 8:24] = torch.tensor([1.0, 0.12, 0.25]).view(3, 1, 1)
        frame[:, 12:20, 15:17] = 1.0
        cues = extract_osu_basic_color_cues(frame)

        self.assertEqual(tuple(cues.shape), (3, 32, 32))
        self.assertGreater(float(cues[0, 10:22, 10:22].mean()), 0.45)
        self.assertLess(float(cues[0, :6, :6].mean()), 0.05)
        self.assertGreater(float(cues[1, 12:20, 15:17].mean()), 0.80)
        self.assertGreater(float(cues[2].max()), 0.10)

    def test_append_color_cues_is_configurable(self) -> None:
        frame = torch.zeros(3, 8, 8)
        self.assertIs(append_color_cues(frame, mode="disabled"), frame)
        augmented = append_color_cues(frame, mode="osu_basic")
        self.assertEqual(
            augmented.shape[0],
            3 + color_cue_channel_count("osu_basic"),
        )

    def test_model_stack_accepts_augmented_input_channels(self) -> None:
        settings = Settings(
            input={"color_cues": "osu_basic"},
            local_encoder={"stem_channels": 4, "feature_channels": 8},
            global_encoder={
                "input_height": 32,
                "input_width": 32,
                "feature_channels": 8,
            },
            fusion={"hidden_dim": 16, "heads": 4, "sampling_points": 2, "layers": 1},
        )
        modules = build_model_stack(settings)
        self.assertEqual(modules["local"].stem[0].in_channels, 6)
        self.assertEqual(modules["global"].stage2.block[0].in_channels, 6)


if __name__ == "__main__":
    unittest.main()
