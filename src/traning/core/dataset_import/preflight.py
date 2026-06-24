from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil

from traning.lib.data import DiscoveryResult, discover_segments
from traning.lib.data.models import DatasetIssue
from traning.conf import DataSplit, Settings


@dataclass(frozen=True)
class DataInputReport:
    split: DataSplit
    segment_count: int
    frame_count_estimate: int
    item_counts: dict[str, int]
    category_counts: dict[str, int]
    dimension_counts: dict[str, int]
    issue_count: int
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.segment_count > 0 and self.issue_count == 0


def _combine_item_filters(
    base_items: tuple[str, ...],
    split_items: tuple[str, ...],
) -> tuple[str, ...]:
    if base_items and split_items:
        return tuple(item for item in base_items if item in set(split_items))
    return split_items or base_items


def _split_items(config, split: DataSplit) -> tuple[str, ...]:
    if split == "train":
        return config.train_items
    if split == "validation":
        return config.validation_items
    return ()


def discover_data_input(
    settings: Settings,
    *,
    split: DataSplit = "all",
) -> DiscoveryResult:
    config = settings.data_input
    split_items = _split_items(config, split)
    if split != "all" and not split_items:
        return DiscoveryResult(
            records=(),
            issues=(
                DatasetIssue(
                    config.dataset_root,
                    f"{split} split has no configured items",
                ),
            ),
        )
    return discover_segments(
        config.dataset_root,
        dimensions=config.dimensions,
        categories=config.categories,
        include_items=_combine_item_filters(config.include_items, split_items),
        exclude_items=config.exclude_items,
        max_segments=config.max_segments,
    )


def inspect_data_input(
    settings: Settings,
    *,
    split: DataSplit = "all",
) -> DataInputReport:
    result = discover_data_input(settings, split=split)
    fps = settings.data_input.sample_fps
    frame_step = settings.data_input.frame_step
    max_frames = settings.data_input.max_frames_per_segment
    estimated_frames = 0

    for record in result.records:
        count = max(1, ceil(record.annotation.duration_ms * fps / 1000.0))
        count = (count + frame_step - 1) // frame_step
        if max_frames is not None:
            count = min(count, max_frames)
        estimated_frames += count

    return DataInputReport(
        split=split,
        segment_count=len(result.records),
        frame_count_estimate=estimated_frames,
        item_counts=dict(Counter(item.item_name for item in result.records)),
        category_counts=dict(Counter(item.category for item in result.records)),
        dimension_counts=dict(
            Counter(item.dataset_dimension for item in result.records)
        ),
        issue_count=len(result.issues),
        issues=tuple(f"{issue.path}: {issue.message}" for issue in result.issues),
    )


__all__ = [
    "DataInputReport",
    "discover_data_input",
    "inspect_data_input",
]
