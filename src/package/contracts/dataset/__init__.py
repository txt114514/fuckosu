from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from package.contracts.base import ContractMixin


class DataSplit(StrEnum):
    ALL = "all"
    TRAIN = "train"
    VALIDATION = "validation"


class DatasetDimension(StrEnum):
    ATOMIC = "atomic"
    LONG_SEQUENCE = "long_sequence"


class SegmentCategory(StrEnum):
    SINGLE_POINT = "single_point"
    SLIDER = "slider"
    MULTI_POINT = "multi_point"
    POINT_SLIDER = "point_slider"
    SPINNER = "spinner"


@dataclass(frozen=True)
class TrainingItemRef(ContractMixin):
    item_name: str
    root: str

    def __post_init__(self) -> None:
        if not self.item_name:
            raise ValueError("item_name must not be empty")
        if not self.root:
            raise ValueError("root must not be empty")

    @classmethod
    def from_path(cls, item_name: str, root: Path) -> TrainingItemRef:
        return cls(item_name=item_name, root=root.as_posix())


@dataclass(frozen=True)
class SegmentRef(ContractMixin):
    sample_key: str
    item_name: str
    category: SegmentCategory
    dimension: DatasetDimension
    video_path: str
    annotation_path: str

    def __post_init__(self) -> None:
        if not self.sample_key or not self.item_name:
            raise ValueError("segment identity fields must not be empty")
        if not isinstance(self.category, SegmentCategory):
            object.__setattr__(self, "category", SegmentCategory(self.category))
        if not isinstance(self.dimension, DatasetDimension):
            object.__setattr__(self, "dimension", DatasetDimension(self.dimension))
        if not self.video_path or not self.annotation_path:
            raise ValueError("segment paths must not be empty")


@dataclass(frozen=True)
class FrameSampleRef(ContractMixin):
    sample_key: str
    frame_index: int
    timestamp_ms: float

    def __post_init__(self) -> None:
        if not self.sample_key:
            raise ValueError("sample_key must not be empty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be nonnegative")


@dataclass(frozen=True)
class SegmentManifestEntry(ContractMixin):
    segment: SegmentRef
    split: DataSplit = DataSplit.ALL
    schema_version: str = "segment-manifest-v1"

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        if not isinstance(self.segment, SegmentRef):
            object.__setattr__(self, "segment", SegmentRef.from_mapping(self.segment))
        if not isinstance(self.split, DataSplit):
            object.__setattr__(self, "split", DataSplit(self.split))


__all__ = [
    "DataSplit",
    "DatasetDimension",
    "FrameSampleRef",
    "SegmentCategory",
    "SegmentManifestEntry",
    "SegmentRef",
    "TrainingItemRef",
]
