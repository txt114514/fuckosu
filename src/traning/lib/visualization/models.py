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
    status: VisualizationStatus
    output_path: Path | None = None
    warning: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in {"saved", "displayed"}


@dataclass(frozen=True)
class GalleryResult:
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
