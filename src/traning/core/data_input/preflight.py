from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil

from traning.Lib.data import DiscoveryResult, discover_segments
from traning.conf import Settings


@dataclass(frozen=True)
class DataInputReport:
    segment_count: int
    frame_count_estimate: int
    category_counts: dict[str, int]
    dimension_counts: dict[str, int]
    issue_count: int
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.segment_count > 0 and self.issue_count == 0


def discover_data_input(settings: Settings) -> DiscoveryResult:
    config = settings.data_input
    return discover_segments(
        config.dataset_root,
        dimensions=config.dimensions,
        categories=config.categories,
        max_segments=config.max_segments,
    )


def inspect_data_input(settings: Settings) -> DataInputReport:
    result = discover_data_input(settings)
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
        segment_count=len(result.records),
        frame_count_estimate=estimated_frames,
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
