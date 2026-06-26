from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from statistics import mean

from package.dataset_split import default_split_manifest_path, load_split_manifest
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
    distribution: dict[str, object] = field(default_factory=dict)

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
    if split == "all":
        return ()
    manifest_path = config.split_manifest_path or default_split_manifest_path(
        config.dataset_root
    )
    manifest = load_split_manifest(manifest_path)
    if manifest is not None:
        return manifest.split_items(split)
    if split == "train":
        return config.train_items
    if split == "validation":
        return config.validation_items
    if split == "test":
        return config.test_items
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
    distribution, data_issues = _distribution_and_topology(result.records)
    distribution["data_quality_issues"] = data_issues

    return DataInputReport(
        split=split,
        segment_count=len(result.records),
        frame_count_estimate=estimated_frames,
        item_counts=dict(Counter(item.item_name for item in result.records)),
        category_counts=dict(Counter(item.category for item in result.records)),
        dimension_counts=dict(
            Counter(item.dataset_dimension for item in result.records)
        ),
        distribution=distribution,
        issue_count=len(result.issues),
        issues=tuple(f"{issue.path}: {issue.message}" for issue in result.issues),
    )


def _distribution_and_topology(records) -> tuple[dict[str, object], tuple[str, ...]]:
    object_counts: Counter[str] = Counter()
    repeat_counts: Counter[int] = Counter()
    spinner_durations: list[float] = []
    slider_durations: list[float] = []
    inter_object_intervals: list[float] = []
    dense_windows = 0
    issues: list[str] = []
    segment_coverage: dict[str, dict[str, object]] = {}
    for record in records:
        objects = tuple(record.annotation.hit_objects)
        sorted_objects = sorted(objects, key=lambda item: (item.start_ms, item.source_index or 0))
        for previous, current in zip(sorted_objects, sorted_objects[1:]):
            inter_object_intervals.append(float(current.start_ms - previous.start_ms))
        for item in sorted_objects:
            kind = _kind(item.type)
            object_counts[kind] += 1
            if kind == "spinner":
                spinner_durations.append(float(item.end_ms - item.start_ms))
            if kind == "slider":
                repeat_counts[int(item.repeats)] += 1
                slider_durations.append(float(item.end_ms - item.start_ms))
                issues.extend(_slider_topology_issues(record, item))
        dense_windows += _high_density_windows(sorted_objects)
        segment_coverage[record.key] = {
            "object_count": len(objects),
            "valid_frame_estimate": max(1, ceil(record.annotation.duration_ms / 1000 * 60)),
            "duration_ms": record.annotation.duration_ms,
        }
    total = sum(object_counts.values())
    distribution = {
        "object_counts": dict(object_counts),
        "object_ratios": {
            key: value / total for key, value in object_counts.items()
        } if total else {},
        "spinner_duration_ms": _summary(spinner_durations),
        "long_sequence_count": sum(1 for record in records if record.dataset_dimension == "long_sequence"),
        "slider_repeat_distribution": dict(repeat_counts),
        "circle_density": {
            "high_density_window_count": dense_windows,
            "min_inter_object_interval_ms": min(inter_object_intervals) if inter_object_intervals else None,
            "mean_inter_object_interval_ms": mean(inter_object_intervals) if inter_object_intervals else None,
        },
        "long_slider_duration_ms": _summary([value for value in slider_durations if value >= 1000.0]),
        "segment_coverage": segment_coverage,
    }
    return distribution, tuple(issues)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {"count": len(values), "min": min(values), "mean": mean(values), "max": max(values)}


def _kind(value: str) -> str:
    raw = value.lower()
    if "spinner" in raw:
        return "spinner"
    if "slider" in raw:
        return "slider"
    return "circle"


def _high_density_windows(objects) -> int:
    count = 0
    for index, item in enumerate(objects):
        window = [other for other in objects[index:] if other.start_ms - item.start_ms <= 500]
        if len(window) >= 6:
            count += 1
    return count


def _slider_topology_issues(record, item) -> tuple[str, ...]:
    issues: list[str] = []
    path = tuple(item.path)
    ident = f"{record.key}:{item.source_index}"
    if item.repeats < 1 or item.repeats > 16:
        issues.append(f"{ident}: abnormal_repeat severity=error blocks_training=true")
    if len(path) < 2:
        issues.append(f"{ident}: degenerate_path severity=error blocks_training=true")
        return tuple(issues)
    if item.end_ms - item.start_ms < 80:
        issues.append(f"{ident}: extremely_short_slider severity=warning blocks_training=false")
    if any(not (0 <= x <= 512 and 0 <= y <= 384) for x, y in path):
        issues.append(f"{ident}: slider_out_of_playfield severity=error blocks_training=true")
    if _self_intersects(path):
        issues.append(f"{ident}: slider_self_intersection severity=error blocks_training=true")
    if _touches_branch(path):
        issues.append(f"{ident}: contact_branch severity=error blocks_training=true")
    return tuple(issues)


def _self_intersects(path: tuple[tuple[float, float], ...]) -> bool:
    segments = tuple(zip(path, path[1:]))
    for index, first in enumerate(segments):
        for other_index, second in enumerate(segments[index + 2 :], start=index + 2):
            if other_index == index + 1:
                continue
            if _segments_intersect(first[0], first[1], second[0], second[1]):
                return True
    return False


def _touches_branch(path: tuple[tuple[float, float], ...]) -> bool:
    seen: dict[tuple[int, int], int] = {}
    for index, (x, y) in enumerate(path):
        key = (round(x * 10), round(y * 10))
        if key in seen and index - seen[key] > 1:
            return True
        seen[key] = index
    return False


def _segments_intersect(a, b, c, d) -> bool:
    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


__all__ = [
    "DataInputReport",
    "discover_data_input",
    "inspect_data_input",
]
