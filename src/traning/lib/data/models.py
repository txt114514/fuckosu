from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from traning.lib.data.annotation import SegmentAnnotation


@dataclass(frozen=True)
class SegmentRecord:
    key: str
    item_name: str
    category: str
    dataset_dimension: str
    directory: Path
    video_path: Path
    annotation_path: Path
    annotation: SegmentAnnotation


@dataclass(frozen=True)
class DatasetIssue:
    path: Path
    message: str


@dataclass(frozen=True)
class DiscoveryResult:
    records: tuple[SegmentRecord, ...]
    issues: tuple[DatasetIssue, ...]


@dataclass(frozen=True)
class FrameReference:
    record_index: int
    frame_index: int
    timestamp_ms: float


__all__ = [
    "DatasetIssue",
    "DiscoveryResult",
    "FrameReference",
    "SegmentRecord",
]
