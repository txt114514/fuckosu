"""提供训练侧可调用的最佳试验图集导出稳定 API。"""

from __future__ import annotations

from pathlib import Path
from importlib import import_module

from traning.lib.data import SegmentFrameDataset
from traning.state.gallery_schema import BatchGalleryRequest


def export_best_trial_gallery(
    dataset: SegmentFrameDataset,
    request: BatchGalleryRequest,
    *,
    output_root: Path,
    samples_per_group: int = 10,
):
    # 图集实现依赖训练 Dataset 和图片库，按需导入可保持普通报告器启动路径轻量。
    save_best_trial_gallery = import_module(
        "visualization.core.gallery.exporter"
    ).save_best_trial_gallery

    return save_best_trial_gallery(
        dataset,
        request,
        output_root=output_root,
        samples_per_group=samples_per_group,
    )


__all__ = ["export_best_trial_gallery"]
