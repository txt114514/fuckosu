"""定义按 item 冻结归属的数据划分清单及比例配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from package.contracts import DataSplit


DATASET_SPLIT_SCHEMA_VERSION = 1
DatasetSplit = DataSplit
"""旧公开名称指向 canonical DataSplit 本身，不再定义第二套类型。"""
SPLIT_ORDER: tuple[DataSplit, ...] = (
    DataSplit.TRAIN,
    DataSplit.VALIDATION,
    DataSplit.TEST,
)


@dataclass(frozen=True)
class SplitRatios:
    """train/validation/test 的相对权重；同步前会归一化，不要求输入和为 1。"""

    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value < 0 or value != value or value == float("inf") for value in values):
            raise ValueError("split ratios must be finite and nonnegative")
        total = sum(values)
        if total <= 0:
            raise ValueError("at least one split ratio must be positive")

    def normalized(self) -> "SplitRatios":
        total = self.train + self.validation + self.test
        return SplitRatios(
            train=self.train / total,
            validation=self.validation / total,
            test=self.test / total,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "SplitRatios":
        if raw is None:
            return cls()
        return cls(
            train=float(raw.get("train", 0.8)),
            validation=float(raw.get("validation", 0.1)),
            test=float(raw.get("test", 0.1)),
        )


@dataclass(frozen=True)
class DatasetSplitItem:
    """一个 item 的冻结归属；segment_count 可刷新，但 split 不随比例重排。"""

    item_name: str
    split: DatasetSplit
    segment_count: int
    assigned_at_utc: str
    assignment_reason: str
    source_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.split, DataSplit):
            object.__setattr__(self, "split", DataSplit(self.split))
        if self.split is DataSplit.ALL:
            raise ValueError("item 不得分配到 all split")

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_name": self.item_name,
            "split": self.split,
            "segment_count": self.segment_count,
            "assigned_at_utc": self.assigned_at_utc,
            "assignment_reason": self.assignment_reason,
            "source_name": self.source_name,
        }

    @classmethod
    def from_mapping(cls, item_name: str, raw: Mapping[str, Any]) -> "DatasetSplitItem":
        split = DataSplit(str(raw["split"]))
        if split is DataSplit.ALL:
            raise ValueError(f"unknown split for {item_name}: {split}")
        return cls(
            item_name=item_name,
            split=split,
            segment_count=int(raw.get("segment_count", 0)),
            assigned_at_utc=str(raw.get("assigned_at_utc", "")),
            assignment_reason=str(raw.get("assignment_reason", "loaded")),
            source_name=(
                str(raw["source_name"])
                if raw.get("source_name") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class DatasetSplitManifest:
    """以 item 为最小单位的版本化划分清单，防止同一谱面跨 split 泄漏。"""

    seed: int
    ratios: SplitRatios
    items: Mapping[str, DatasetSplitItem]
    unit: str = "item"
    schema_version: int = DATASET_SPLIT_SCHEMA_VERSION
    allow_test_growth: bool = False

    def split_items(self, split: DatasetSplit) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.item_name
                for item in self.items.values()
                if item.split == split
            )
        )

    def counts(self) -> dict[str, int]:
        return {
            split: len(self.split_items(split))
            for split in SPLIT_ORDER
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "unit": self.unit,
            "ratios": self.ratios.as_dict(),
            "allow_test_growth": self.allow_test_growth,
            "items": {
                item_name: item.as_dict()
                for item_name, item in sorted(self.items.items())
            },
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetSplitManifest":
        items_raw = raw.get("items", {})
        if not isinstance(items_raw, Mapping):
            raise ValueError("split manifest items must be an object")
        items = {
            str(item_name): DatasetSplitItem.from_mapping(str(item_name), item)
            for item_name, item in items_raw.items()
            if isinstance(item, Mapping)
        }
        return cls(
            schema_version=int(raw.get("schema_version", DATASET_SPLIT_SCHEMA_VERSION)),
            seed=int(raw.get("seed", 2026)),
            unit=str(raw.get("unit", "item")),
            ratios=SplitRatios.from_mapping(raw.get("ratios")),
            allow_test_growth=bool(raw.get("allow_test_growth", False)),
            items=items,
        )


@dataclass(frozen=True)
class DatasetSplitSyncResult:
    """一次增量同步的结果；dry_run 时 manifest 是预测结果但磁盘未更新。"""

    manifest_path: Path
    created: bool
    changed: bool
    dry_run: bool
    new_items: tuple[DatasetSplitItem, ...]
    manifest: DatasetSplitManifest

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "created": self.created,
            "changed": self.changed,
            "dry_run": self.dry_run,
            "new_items": tuple(item.as_dict() for item in self.new_items),
            "split_counts": self.manifest.counts(),
            "manifest": self.manifest.as_dict(),
        }


__all__ = [
    "DATASET_SPLIT_SCHEMA_VERSION",
    "DatasetSplit",
    "DatasetSplitItem",
    "DatasetSplitManifest",
    "DatasetSplitSyncResult",
    "SPLIT_ORDER",
    "SplitRatios",
]
