"""发现 segment，并构建携带逐记录坐标规格的 Dataset/DataLoader。"""

from __future__ import annotations

from functools import partial
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from traning.lib.coordinates import transform_from_settings_or_sample
from traning.lib.data import SegmentFrameDataset, collate_frame_samples
from traning.conf import DataSplit, Settings
from traning.core.dataset_import.preflight import discover_data_input


def build_dataset(
    settings: Settings,
    *,
    split: DataSplit = "train",
) -> SegmentFrameDataset:
    """构建 split Dataset，并为每条 record 固化其完整坐标变换规格。"""

    result = discover_data_input(settings, split=split)
    if settings.data_input.strict and result.issues:
        details = "\n".join(
            f"- {issue.path}: {issue.message}" for issue in result.issues[:20]
        )
        raise ValueError(f"data input validation failed:\n{details}")
    if not result.records:
        raise ValueError(f"no {split} segments matched the data input filters")

    config = settings.data_input
    coordinate_transforms = {}
    for record in result.records:
        # 预处理元数据可能随 record 不同，因此不能只为整个 Dataset 解析一次。
        _, spec = transform_from_settings_or_sample(
            settings,
            {"preprocessing_metadata": record.preprocessing_metadata},
            frame_width=settings.input.width,
            frame_height=settings.input.height,
        )
        # 保存完整 spec（包括 affine matrix），使无 Settings 的 gallery 也能复现映射。
        coordinate_transforms[record.key] = spec.as_dict()
    return SegmentFrameDataset(
        result.records,
        sample_fps=config.sample_fps,
        frame_step=config.frame_step,
        max_frames_per_segment=config.max_frames_per_segment,
        visibility_post_ms=config.visibility_post_ms,
        normalize_images=config.normalize_images,
        coordinate_transforms=coordinate_transforms,
    )


def build_dataloader(
    settings: Settings,
    *,
    split: DataSplit = "train",
    shuffle: bool | None = None,
) -> DataLoader:
    """用确定性随机种子和项目配置包装 Dataset。"""

    loader = settings.loader
    generator = torch.Generator()
    generator.manual_seed(int(settings.runtime.seed))
    worker_options = {}
    if loader.num_workers > 0:
        worker_options["persistent_workers"] = loader.persistent_workers
        worker_options["worker_init_fn"] = partial(
            _seed_worker,
            base_seed=int(settings.runtime.seed),
        )
        if loader.prefetch_factor is not None:
            worker_options["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(
        build_dataset(settings, split=split),
        batch_size=loader.batch_size,
        shuffle=loader.shuffle if shuffle is None else shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        generator=generator,
        collate_fn=collate_frame_samples,
        **worker_options,
    )


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


__all__ = ["build_dataloader", "build_dataset"]
