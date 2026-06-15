from __future__ import annotations

from pathlib import Path

from traning.Lib.data.annotation import load_annotation
from traning.Lib.data.models import DatasetIssue, DiscoveryResult, SegmentRecord


def discover_segments(
    dataset_root: Path,
    *,
    dimensions: tuple[str, ...] = (),
    categories: tuple[str, ...] = (),
    max_segments: int | None = None,
) -> DiscoveryResult:
    records: list[SegmentRecord] = []
    issues: list[DatasetIssue] = []
    dimension_filter = set(dimensions)
    category_filter = set(categories)

    if not dataset_root.is_dir():
        return DiscoveryResult(
            records=(),
            issues=(DatasetIssue(dataset_root, "dataset root does not exist"),),
        )

    for annotation_path in sorted(dataset_root.glob("item_*/*/*/beatmap.json")):
        segment_directory = annotation_path.parent
        video_path = segment_directory / "video.mp4"
        try:
            annotation = load_annotation(annotation_path)
        except (ValueError, TypeError) as error:
            issues.append(DatasetIssue(annotation_path, str(error)))
            continue

        if dimension_filter and annotation.dataset_dimension not in dimension_filter:
            continue
        if category_filter and annotation.category not in category_filter:
            continue
        if not video_path.is_file():
            issues.append(DatasetIssue(video_path, "paired video.mp4 is missing"))
            continue

        item_name = annotation_path.parents[2].name
        records.append(
            SegmentRecord(
                key=f"{item_name}/{annotation.segment_id}",
                item_name=item_name,
                category=annotation.category,
                dataset_dimension=annotation.dataset_dimension,
                directory=segment_directory,
                video_path=video_path,
                annotation_path=annotation_path,
                annotation=annotation,
            )
        )
        if max_segments is not None and len(records) >= max_segments:
            break

    if not records and not issues:
        issues.append(DatasetIssue(dataset_root, "no segment annotations were found"))
    return DiscoveryResult(tuple(records), tuple(issues))


__all__ = ["discover_segments"]
