from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import torch

from traning.Lib.training import SliderPathCandidate
from traning.Lib.training.spatial_decode import SpatialCandidate
from traning.conf import DataSplit, Settings
from traning.core.dataset_import import build_dataset
from traning.core.spatial_training import (
    run_spatial_frame_inference,
    slider_path_to_dict,
    spatial_candidate_to_dict,
)


CANDIDATE_CACHE_VERSION = "spatial-candidate-cache-v1"


@dataclass(frozen=True)
class CandidateCacheBuildResult:
    output_dir: Path
    manifest_path: Path
    records_path: Path
    device: str
    split: str
    frames: int
    candidates: int
    slider_paths: int
    ambiguous_candidates: int
    ambiguous_slider_paths: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "records_path": self.records_path,
            "device": self.device,
            "split": self.split,
            "frames": self.frames,
            "candidates": self.candidates,
            "slider_paths": self.slider_paths,
            "ambiguous_candidates": self.ambiguous_candidates,
            "ambiguous_slider_paths": self.ambiguous_slider_paths,
        }


def generate_candidate_cache(
    settings: Settings,
    *,
    output_dir: Path,
    device: torch.device,
    split: DataSplit = "train",
    max_frames: int | None = None,
    patch_limit: int | None = None,
    max_candidates: int | None = None,
    score_threshold: float | None = None,
    nms_radius_px: float | None = None,
    slider_threshold: float | None = None,
    max_slider_paths: int | None = None,
    dataset: Sequence[Mapping[str, Any]] | None = None,
) -> CandidateCacheBuildResult:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when set")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "frames.jsonl"
    manifest_path = output_dir / "manifest.json"

    cache = settings.candidate_cache
    selected_max_candidates = max_candidates or cache.max_candidates_per_frame
    selected_score_threshold = (
        cache.score_threshold if score_threshold is None else score_threshold
    )
    selected_nms_radius = (
        cache.nms_radius_px if nms_radius_px is None else nms_radius_px
    )
    selected_slider_threshold = (
        cache.slider_threshold if slider_threshold is None else slider_threshold
    )
    selected_max_slider_paths = max_slider_paths or cache.max_slider_paths

    source = dataset if dataset is not None else build_dataset(settings, split=split)
    frame_total = len(source) if max_frames is None else min(len(source), max_frames)
    if frame_total <= 0:
        raise ValueError("candidate cache dataset must not be empty")

    total_candidates = 0
    total_slider_paths = 0
    ambiguous_candidates = 0
    ambiguous_slider_paths = 0
    with records_path.open("w", encoding="utf-8") as handle:
        for index in range(frame_total):
            sample = source[index]
            result = run_spatial_frame_inference(
                settings,
                sample,
                device=device,
                max_candidates=selected_max_candidates,
                score_threshold=selected_score_threshold,
                nms_radius_px=selected_nms_radius,
                slider_threshold=selected_slider_threshold,
                max_slider_paths=selected_max_slider_paths,
                slider_min_cells=cache.slider_min_cells,
                slider_path_points=cache.slider_path_points,
                patch_limit=patch_limit,
            )
            record = build_candidate_cache_record(
                sample,
                result.candidates,
                result.slider_paths,
                frame_width=int(sample["image"].shape[-1]),
                frame_height=int(sample["image"].shape[-2]),
                device=str(device),
                patches_processed=result.patches_processed,
                frame_channels=result.frame_channels,
                save_dtype=cache.save_dtype,
                low_confidence_threshold=cache.low_confidence_threshold,
                close_score_margin=cache.close_score_margin,
                slider_attach_distance_px=cache.slider_attach_distance_px,
            )
            total_candidates += len(record["candidates"])
            total_slider_paths += len(record["slider_paths"])
            ambiguous_candidates += sum(
                1 for candidate in record["candidates"] if candidate["ambiguous"]
            )
            ambiguous_slider_paths += sum(
                1 for path in record["slider_paths"] if path["ambiguous"]
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "version": CANDIDATE_CACHE_VERSION,
        "split": split,
        "device": str(device),
        "frames": frame_total,
        "records": str(records_path.name),
        "max_candidates_per_frame": selected_max_candidates,
        "score_threshold": selected_score_threshold,
        "nms_radius_px": selected_nms_radius,
        "slider_threshold": selected_slider_threshold,
        "max_slider_paths": selected_max_slider_paths,
        "save_dtype": cache.save_dtype,
        "candidate_count": total_candidates,
        "slider_path_count": total_slider_paths,
        "ambiguous_candidate_count": ambiguous_candidates,
        "ambiguous_slider_path_count": ambiguous_slider_paths,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return CandidateCacheBuildResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        records_path=records_path,
        device=str(device),
        split=str(split),
        frames=frame_total,
        candidates=total_candidates,
        slider_paths=total_slider_paths,
        ambiguous_candidates=ambiguous_candidates,
        ambiguous_slider_paths=ambiguous_slider_paths,
    )


def build_candidate_cache_record(
    sample: Mapping[str, Any],
    candidates: Sequence[SpatialCandidate],
    slider_paths: Sequence[SliderPathCandidate],
    *,
    frame_width: int,
    frame_height: int,
    device: str,
    patches_processed: int,
    frame_channels: int,
    save_dtype: str,
    low_confidence_threshold: float,
    close_score_margin: float,
    slider_attach_distance_px: float,
) -> dict[str, Any]:
    slider_rows = [slider_path_to_dict(path) for path in slider_paths]
    candidate_rows = []
    for index, candidate in enumerate(candidates):
        attached = _nearest_slider_path(
            candidate,
            slider_paths,
            max_distance=slider_attach_distance_px,
        )
        reasons = _candidate_ambiguity_reasons(
            index,
            candidates,
            attached,
            low_confidence_threshold=low_confidence_threshold,
            close_score_margin=close_score_margin,
        )
        row = spatial_candidate_to_dict(candidate)
        row.update(
            {
                "candidate_id": index,
                "embedding": _cast_embedding(candidate.embedding, save_dtype),
                "slider_path_id": (None if attached is None else attached.component_id),
                "ambiguous": bool(reasons),
                "ambiguity_reasons": reasons,
            }
        )
        candidate_rows.append(row)
    return {
        "version": CANDIDATE_CACHE_VERSION,
        "sample_key": sample.get("sample_key"),
        "frame_index": sample.get("frame_index"),
        "timestamp_ms": sample.get("timestamp_ms"),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "device": device,
        "patches_processed": patches_processed,
        "frame_channels": frame_channels,
        "candidates": candidate_rows,
        "slider_paths": slider_rows,
    }


def _candidate_ambiguity_reasons(
    index: int,
    candidates: Sequence[SpatialCandidate],
    slider_path: SliderPathCandidate | None,
    *,
    low_confidence_threshold: float,
    close_score_margin: float,
) -> tuple[str, ...]:
    candidate = candidates[index]
    reasons: list[str] = []
    if candidate.score < low_confidence_threshold:
        reasons.append("low_confidence")
    if _has_close_neighbor(index, candidates, margin=close_score_margin):
        reasons.append("close_score")
    if candidate.object_type.startswith("slider"):
        if slider_path is None:
            reasons.append("missing_slider_path")
        elif slider_path.ambiguous:
            reasons.append("slider_path_ambiguous")
    return tuple(reasons)


def _has_close_neighbor(
    index: int,
    candidates: Sequence[SpatialCandidate],
    *,
    margin: float,
) -> bool:
    if margin <= 0:
        return False
    score = candidates[index].score
    return any(
        abs(score - candidate.score) <= margin
        for other, candidate in enumerate(candidates)
        if other != index
    )


def _nearest_slider_path(
    candidate: SpatialCandidate,
    paths: Sequence[SliderPathCandidate],
    *,
    max_distance: float,
) -> SliderPathCandidate | None:
    if not candidate.object_type.startswith("slider") or not paths:
        return None
    best_path: SliderPathCandidate | None = None
    best_distance = float("inf")
    for path in paths:
        distance = _distance_to_polyline((candidate.x, candidate.y), path.polyline)
        if distance < best_distance:
            best_distance = distance
            best_path = path
    if best_distance <= max_distance:
        return best_path
    return None


def _distance_to_polyline(
    point: tuple[float, float],
    polyline: Sequence[tuple[float, float]],
) -> float:
    if not polyline:
        return float("inf")
    if len(polyline) == 1:
        return _point_distance(point, polyline[0])
    return min(
        _point_to_segment_distance(point, start, end)
        for start, end in zip(polyline, polyline[1:])
    )


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    vx = bx - ax
    vy = by - ay
    length_squared = vx * vx + vy * vy
    if length_squared <= 1e-6:
        return _point_distance(point, start)
    t = ((px - ax) * vx + (py - ay) * vy) / length_squared
    t = min(max(t, 0.0), 1.0)
    return _point_distance(point, (ax + t * vx, ay + t * vy))


def _point_distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _cast_embedding(values: Sequence[float], save_dtype: str) -> list[float]:
    tensor = torch.tensor(tuple(values), dtype=torch.float32)
    if save_dtype == "float16":
        tensor = tensor.to(torch.float16).to(torch.float32)
    elif save_dtype != "float32":
        raise ValueError(f"unsupported candidate cache dtype: {save_dtype}")
    return [float(value) for value in tensor.tolist()]


__all__ = [
    "CANDIDATE_CACHE_VERSION",
    "CandidateCacheBuildResult",
    "build_candidate_cache_record",
    "generate_candidate_cache",
]
