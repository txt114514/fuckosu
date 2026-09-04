"""数据发现和帧采样阶段使用的不可变记录类型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from traning.state.common import JSONObject
from traning.lib.data.annotation import SegmentAnnotation


@dataclass(frozen=True)
class SegmentRecord:
    """一个已配对且通过标注解析的 segment 数据记录。"""

    key: str
    item_name: str
    category: str
    dataset_dimension: str
    directory: Path
    video_path: Path
    annotation_path: Path
    annotation: SegmentAnnotation
    preprocessing_metadata: JSONObject | None = None


@dataclass(frozen=True)
class DatasetIssue:
    """数据发现时可定位但不隐式吞掉的文件问题。"""

    path: Path
    message: str


@dataclass(frozen=True)
class DiscoveryResult:
    """数据发现得到的有效记录与全部非致命问题。"""

    records: tuple[SegmentRecord, ...]
    issues: tuple[DatasetIssue, ...]


@dataclass(frozen=True)
class FrameReference:
    """稳定指向某个 segment 中一个采样帧的轻量引用。"""

    record_index: int
    frame_index: int
    timestamp_ms: float


__all__ = [
    "DatasetIssue",
    "DiscoveryResult",
    "FrameReference",
    "SegmentRecord",
]
