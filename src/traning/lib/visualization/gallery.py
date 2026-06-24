from __future__ import annotations

import json
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from traning.lib.data import SegmentFrameDataset, SegmentRecord
from traning.lib.visualization.output_identity import allocate_output_identity
from traning.lib.visualization.render import (
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
    grouped: dict[tuple[bool, str], list[FrameEvaluation]] = defaultdict(list)
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
        grouped[(frame.passed, subproject)].append(frame)

    rng = random.Random(request.random_seed)
    selected: dict[tuple[bool, str], tuple[FrameEvaluation, ...]] = {}
    for key, frames in grouped.items():
        count = min(samples_per_group, len(frames))
        selected[key] = tuple(rng.sample(frames, count))

    working_dir.mkdir(parents=True, exist_ok=True)
    reached_subprojects = {
        subproject for _, subproject in grouped
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
    for passed in (True, False):
        for subproject in EVALUATION_SUBPROJECTS:
            frames = selected.get((passed, subproject), ())
            if not frames:
                continue
            destination = working_dir / OUTCOME_DIRECTORIES[passed] / subproject
            for sequence, frame in enumerate(frames, start=1):
                dataset_index, _ = lookup[(frame.sample_key, frame.frame_index)]
                sample = dataset[dataset_index]
                metadata = (
                    f"output={output_identity.sequence:06d}",
                    f"output_time={output_identity.created_at_utc}",
                    f"batch={request.batch_id}",
                    f"trial={best_trial.trial_id} score={best_trial.score:.6g}",
                    f"score_version={best_trial.score_version}",
                    f"subproject={subproject}",
                    f"outcome={OUTCOME_DIRECTORIES[passed]}",
                    *_metric_lines(frame.metrics),
                )
                image = render_annotated_frame(
                    sample,
                    target_source_index=frame.target_source_index,
                    predicted_osu_xy=frame.predicted_osu_xy,
                    metadata_lines=metadata,
                )
                filename = (
                    f"{sequence:02d}__{_safe_name(frame.sample_key)}"
                    f"__frame_{frame.frame_index:06d}.png"
                )
                output_path = save_annotated_frame(image, destination / filename)
                saved_frames.append(
                    {
                        "sample_key": frame.sample_key,
                        "frame_index": frame.frame_index,
                        "passed": frame.passed,
                        "subproject": subproject,
                        "path": str(output_path.relative_to(working_dir)),
                        "target_source_index": frame.target_source_index,
                        "predicted_osu_xy": frame.predicted_osu_xy,
                        "metrics": frame.metrics,
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
                "random_seed": request.random_seed,
                "samples_per_group": samples_per_group,
                "subprojects": list(EVALUATION_SUBPROJECTS),
                "reached_subprojects": [
                    subproject
                    for subproject in EVALUATION_SUBPROJECTS
                    if subproject in reached_subprojects
                ],
                "saved_frame_count": len(saved_frames),
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
