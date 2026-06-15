from __future__ import annotations

from typing import Any

import torch


def collate_frame_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")
    images = [sample["image"] for sample in samples]
    same_shape = len({tuple(image.shape) for image in images}) == 1
    return {
        "images": torch.stack(images) if same_shape else images,
        "samples": tuple(
            {key: value for key, value in sample.items() if key != "image"}
            for sample in samples
        ),
    }


__all__ = ["collate_frame_samples"]
