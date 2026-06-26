from __future__ import annotations

import json
import random
import re
import shutil
import csv
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traning.lib.data import SegmentFrameDataset, SegmentRecord
from visualization.core.gallery.manifest import allocate_output_identity
from visualization.core.gallery.renderer import (
    render_annotated_frame,
    save_annotated_frame,
)
from traning.state.gallery_schema import (
    BatchGalleryRequest,
    EVALUATION_SUBPROJECTS,
    FrameEvaluation,
)


OUTCOME_DIRECTORIES = {
    True: "passed",
    False: "failed",
}
DIMENSION_SUBPROJECTS = {
    "long_sequence": "long_sequence",
}


@dataclass
class _SampleFrameGroup:
    sample_key: str
    subproject: str
    passed: bool = True
    frames: list[FrameEvaluation] = field(default_factory=list)

    def add(self, frame: FrameEvaluation) -> None:
        self.frames.append(frame)
        self.passed = self.passed and frame.passed


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "unnamed"


def _subproject_for_record(record: SegmentRecord) -> str:
    return DIMENSION_SUBPROJECTS.get(
        record.dataset_dimension,
        record.category,
    )


def _frame_lookup(
    dataset: SegmentFrameDataset,
) -> dict[tuple[str, int], tuple[int, str]]:
    return {
        (
            dataset.records[reference.record_index].key,
            reference.frame_index,
        ): (
            dataset_index,
            _subproject_for_record(dataset.records[reference.record_index]),
        )
        for dataset_index, reference in enumerate(dataset.references)
    }


def _metric_lines(metrics: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(
        f"{name}={value:.6g}"
        for name, value in sorted(metrics.items())
    )


def _is_export_frame(frame: FrameEvaluation) -> bool:
    return (
        frame.target_source_index is not None
        or frame.predicted_osu_xy is not None
        or frame.predicted_video_xy is not None
        or frame.primary_error != "none"
    )


def _frame_identity(frame: FrameEvaluation) -> tuple[Any, ...]:
    return (
        frame.frame_index,
        frame.target_source_index,
        frame.predicted_osu_xy,
        frame.predicted_video_xy,
        frame.primary_error,
        frame.error_tags,
    )


def _frame_order_key(frame: FrameEvaluation) -> tuple[Any, ...]:
    return (
        frame.frame_index,
        frame.target_source_index is None,
        frame.target_source_index if frame.target_source_index is not None else -1,
        str(frame.predicted_video_xy),
        str(frame.predicted_osu_xy),
        frame.primary_error,
        frame.error_tags,
    )


def _sample_group_key(group: _SampleFrameGroup) -> tuple[str, str]:
    return group.subproject, group.sample_key


def _sorted_sample_groups(
    groups: list[_SampleFrameGroup],
) -> tuple[_SampleFrameGroup, ...]:
    return tuple(sorted(groups, key=_sample_group_key))


def save_best_trial_gallery(
    dataset: SegmentFrameDataset,
    request: BatchGalleryRequest,
    *,
    output_root: Path,
    samples_per_group: int = 10,
) -> tuple[Path, int, tuple[str, ...]]:
    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive")

    best_trial = request.best_trial
    output_identity = allocate_output_identity(output_root)
    gallery_dir = (
        output_root
        / (
            f"{output_identity.prefix}__{_safe_name(request.batch_id)}"
            f"__{_safe_name(best_trial.trial_id)}"
        )
    )
    working_dir = gallery_dir.with_name(f".{gallery_dir.name}.tmp")
    if working_dir.exists():
        shutil.rmtree(working_dir)
    lookup = _frame_lookup(dataset)
    grouped_by_sample: dict[tuple[str, str], _SampleFrameGroup] = {}
    seen_frame_records: set[tuple[str, str, tuple[Any, ...]]] = set()
    issues: list[str] = []

    for frame in best_trial.frames:
        resolved = lookup.get((frame.sample_key, frame.frame_index))
        if resolved is None:
            issues.append(
                f"missing dataset frame {frame.sample_key}:{frame.frame_index}"
            )
            continue
        _, subproject = resolved
        if subproject not in EVALUATION_SUBPROJECTS:
            issues.append(
                f"unsupported subproject {subproject!r} for "
                f"{frame.sample_key}:{frame.frame_index}"
            )
            continue
        if not _is_export_frame(frame):
            continue
        seen_key = (frame.sample_key, subproject, _frame_identity(frame))
        if seen_key in seen_frame_records:
            continue
        seen_frame_records.add(seen_key)
        sample_key = (frame.sample_key, subproject)
        group = grouped_by_sample.get(sample_key)
        if group is None:
            group = _SampleFrameGroup(
                sample_key=frame.sample_key,
                subproject=subproject,
            )
            grouped_by_sample[sample_key] = group
        group.add(frame)

    rng = random.Random(request.random_seed)
    sample_groups: dict[tuple[bool, str], list[_SampleFrameGroup]] = defaultdict(list)
    for group in grouped_by_sample.values():
        if not group.frames:
            continue
        group.frames.sort(key=_frame_order_key)
        sample_groups[(group.passed, group.subproject)].append(group)

    selected: dict[tuple[bool, str], tuple[_SampleFrameGroup, ...]] = {}
    for key, groups in sample_groups.items():
        ordered_groups = _sorted_sample_groups(groups)
        count = min(samples_per_group, len(ordered_groups))
        selected[key] = tuple(rng.sample(ordered_groups, count))

    working_dir.mkdir(parents=True, exist_ok=True)
    reached_subprojects = {
        subproject for _, subproject in sample_groups
    }
    for passed in (True, False):
        for subproject in EVALUATION_SUBPROJECTS:
            if subproject in reached_subprojects:
                (
                    working_dir
                    / OUTCOME_DIRECTORIES[passed]
                    / subproject
                ).mkdir(parents=True, exist_ok=True)

    saved_frames: list[dict[str, Any]] = []
    saved_sample_groups: list[dict[str, Any]] = []
    for passed in (True, False):
        for subproject in EVALUATION_SUBPROJECTS:
            groups = selected.get((passed, subproject), ())
            if not groups:
                continue
            destination = working_dir / OUTCOME_DIRECTORIES[passed] / subproject
            for group_index, group in enumerate(groups, start=1):
                group_dir = destination / (
                    f"{group_index:02d}__{_safe_name(group.sample_key)}"
                )
                group_dir.mkdir(parents=True, exist_ok=True)
                group_frame_records: list[dict[str, Any]] = []
                for frame_index, frame in enumerate(group.frames, start=1):
                    dataset_index, _ = lookup[(frame.sample_key, frame.frame_index)]
                    sample = dataset[dataset_index]
                    metadata = (
                        f"output={output_identity.sequence:06d}",
                        f"output_time={output_identity.created_at_utc}",
                        f"batch={request.batch_id}",
                        f"trial={best_trial.trial_id} score={best_trial.score:.6g}",
                        f"score_version={best_trial.score_version}",
                        f"sample_group={group_index:02d}",
                        f"subproject={subproject}",
                        f"outcome={OUTCOME_DIRECTORIES[passed]}",
                        *_metric_lines(frame.metrics),
                    )
                    image = render_annotated_frame(
                        sample,
                        target_source_index=frame.target_source_index,
                        predicted_osu_xy=frame.predicted_osu_xy,
                        predicted_video_xy=frame.predicted_video_xy,
                        metadata_lines=metadata,
                    )
                    filename = f"{frame_index:02d}__frame_{frame.frame_index:06d}.png"
                    output_path = save_annotated_frame(image, group_dir / filename)
                    frame_record = {
                        "sample_key": frame.sample_key,
                        "frame_index": frame.frame_index,
                        "passed": frame.passed,
                        "subproject": subproject,
                        "path": str(output_path.relative_to(working_dir)),
                        "target_source_index": frame.target_source_index,
                        "predicted_osu_xy": frame.predicted_osu_xy,
                        "predicted_video_xy": frame.predicted_video_xy,
                        "primary_error": frame.primary_error,
                        "error_tags": frame.error_tags,
                        "frequency_limited": frame.frequency_limited,
                        "metrics": frame.metrics,
                    }
                    saved_frames.append(frame_record)
                    group_frame_records.append(frame_record)
                saved_sample_groups.append(
                    {
                        "sample_key": group.sample_key,
                        "passed": group.passed,
                        "subproject": subproject,
                        "path": str(group_dir.relative_to(working_dir)),
                        "frame_count": len(group_frame_records),
                        "frames": group_frame_records,
                    }
                )

    parameters_path = working_dir / "best_parameters.json"
    parameters_path.write_text(
        json.dumps(
            {
                "batch_id": request.batch_id,
                "output_sequence": output_identity.sequence,
                "output_time_utc": output_identity.created_at_utc,
                "trial_id": best_trial.trial_id,
                "score": best_trial.score,
                "score_version": best_trial.score_version,
                "metrics": best_trial.metrics,
                "parameters": best_trial.parameters.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = working_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "error_type",
                "segment",
                "beatmap",
                "sample",
                "trial",
                "curriculum_stage",
                "parameter_group",
                "score",
                "gallery_image_path",
                "predicted_osu_xy",
                "predicted_video_xy",
                "action_type",
                "ambiguity_reason",
                "score_version",
                "dataset_version",
                "evaluation_dataset_version",
                "candidate_cache_version",
                "transform_version",
                "configuration_version",
            ),
        )
        writer.writeheader()
        metadata = request.metadata
        for frame in saved_frames:
            sample_key = str(frame["sample_key"])
            writer.writerow(
                {
                    "error_type": frame.get("primary_error") or "",
                    "segment": sample_key.rsplit("/", 1)[-1],
                    "beatmap": sample_key.split("/", 1)[0],
                    "sample": sample_key,
                    "trial": best_trial.trial_id,
                    "curriculum_stage": metadata.get("curriculum_stage", ""),
                    "parameter_group": metadata.get("trial_id", best_trial.trial_id),
                    "score": best_trial.score,
                    "gallery_image_path": frame["path"],
                    "predicted_osu_xy": frame.get("predicted_osu_xy"),
                    "predicted_video_xy": frame.get("predicted_video_xy"),
                    "action_type": frame.get("action") or "",
                    "ambiguity_reason": frame.get("ambiguity_reason") or "",
                    "score_version": best_trial.score_version,
                    "dataset_version": metadata.get("dataset_version", ""),
                    "evaluation_dataset_version": metadata.get(
                        "evaluation_dataset_version", ""
                    ),
                    "candidate_cache_version": metadata.get(
                        "candidate_cache_version", ""
                    ),
                    "transform_version": metadata.get("transform_version", ""),
                    "configuration_version": metadata.get("configuration_version", ""),
                }
            )
    manifest_path = working_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch_id": request.batch_id,
                "output_sequence": output_identity.sequence,
                "output_time_utc": output_identity.created_at_utc,
                "selected_trial_id": best_trial.trial_id,
                "selected_trial_score": best_trial.score,
                "score_version": best_trial.score_version,
                "metadata": request.metadata,
                "index_csv": str(index_path.relative_to(working_dir)),
                "random_seed": request.random_seed,
                "samples_per_group": samples_per_group,
                "selected_sample_group_count": len(saved_sample_groups),
                "subprojects": list(EVALUATION_SUBPROJECTS),
                "reached_subprojects": [
                    subproject
                    for subproject in EVALUATION_SUBPROJECTS
                    if subproject in reached_subprojects
                ],
                "saved_frame_count": len(saved_frames),
                "sample_groups": saved_sample_groups,
                "frames": saved_frames,
                "issues": issues,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    working_dir.replace(gallery_dir)
    return gallery_dir, len(saved_frames), tuple(issues)


__all__ = [
    "OUTCOME_DIRECTORIES",
    "save_best_trial_gallery",
]
