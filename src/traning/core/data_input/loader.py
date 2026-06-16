from __future__ import annotations

from torch.utils.data import DataLoader

from traning.Lib.data import SegmentFrameDataset, collate_frame_samples
from traning.conf import DataSplit, Settings
from traning.core.data_input.preflight import discover_data_input


def build_dataset(
    settings: Settings,
    *,
    split: DataSplit = "train",
) -> SegmentFrameDataset:
    result = discover_data_input(settings, split=split)
    if settings.data_input.strict and result.issues:
        details = "\n".join(
            f"- {issue.path}: {issue.message}" for issue in result.issues[:20]
        )
        raise ValueError(f"data input validation failed:\n{details}")
    if not result.records:
        raise ValueError(f"no {split} segments matched the data input filters")

    config = settings.data_input
    return SegmentFrameDataset(
        result.records,
        sample_fps=config.sample_fps,
        frame_step=config.frame_step,
        max_frames_per_segment=config.max_frames_per_segment,
        visibility_post_ms=config.visibility_post_ms,
        normalize_images=config.normalize_images,
    )


def build_dataloader(
    settings: Settings,
    *,
    split: DataSplit = "train",
    shuffle: bool | None = None,
) -> DataLoader:
    loader = settings.loader
    return DataLoader(
        build_dataset(settings, split=split),
        batch_size=loader.batch_size,
        shuffle=loader.shuffle if shuffle is None else shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        collate_fn=collate_frame_samples,
    )


__all__ = ["build_dataloader", "build_dataset"]
