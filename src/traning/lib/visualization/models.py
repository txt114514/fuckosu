"""可视化执行结果、gallery 结果和选中帧的不可变契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


VisualizationStatus = Literal[
    "disabled",
    "skipped",
    "saved",
    "displayed",
    "failed",
]


@dataclass(frozen=True)
class VisualizationResult:
    """单帧保存/显示结果；warning 用于可恢复的可视化失败。"""

    status: VisualizationStatus
    output_path: Path | None = None
    warning: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in {"saved", "displayed"}


@dataclass(frozen=True)
class GalleryResult:
    """gallery 导出结果及实际保存帧数。"""

    status: VisualizationStatus
    output_dir: Path | None = None
    selected_trial_id: str | None = None
    saved_frame_count: int = 0
    warning: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in {"saved", "displayed"}


@dataclass(frozen=True)
class SelectedFrame:
    """用户物件索引解析到的 Dataset 帧引用。"""

    dataset_index: int
    segment_index: int
    object_index: int
    timestamp_ms: float
    target_source_index: int | None


__all__ = [
    "GalleryResult",
    "SelectedFrame",
    "VisualizationResult",
    "VisualizationStatus",
]
