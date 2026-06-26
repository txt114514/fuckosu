from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import torch

from traning.lib.training import SliderPathCandidate
from traning.lib.training.spatial_decode import SpatialCandidate
from traning.conf import DataSplit, Settings
from traning.core.dataset_import import build_dataset
from traning.core.spatial import (
    run_spatial_frame_inference,
    slider_path_to_dict,
    spatial_candidate_to_dict,
)
from traning.lib.coordinates import transform_from_settings_or_sample
from traning.state.versioning import version_manifest


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
                action_window_ms=max(1000.0 / settings.data_input.sample_fps / 2, 1.0),
                settings=settings,
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
        "versions": version_manifest(settings)
        | {"candidate_cache_version": CANDIDATE_CACHE_VERSION},
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
    action_window_ms: float = 25.0,
    settings: Settings | None = None,
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
    _apply_candidate_reviews(
        candidate_rows,
        frame_width=frame_width,
        frame_height=frame_height,
        enabled=bool(
            settings is not None and settings.candidate_cache.ambiguity_review_enabled
        ),
        max_candidates=(
            settings.candidate_cache.ambiguity_review_max_candidates
            if settings is not None
            else 0
        ),
    )
    _apply_local_refinement(
        candidate_rows,
        frame_width=frame_width,
        frame_height=frame_height,
        enabled=bool(
            settings is not None and settings.candidate_cache.local_refiner_enabled
        ),
        top_k=(
            settings.candidate_cache.local_refiner_top_k if settings is not None else 0
        ),
        radius_px=(
            settings.candidate_cache.local_refiner_radius_px
            if settings is not None
            else 0.0
        ),
    )
    _, transform_spec = transform_from_settings_or_sample(
        settings,
        sample,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return {
        "version": CANDIDATE_CACHE_VERSION,
        "coordinate_transform": transform_spec.as_dict(),
        "sample_key": sample.get("sample_key"),
        "frame_index": sample.get("frame_index"),
        "timestamp_ms": sample.get("timestamp_ms"),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "device": device,
        "patches_processed": patches_processed,
        "frame_channels": frame_channels,
        "temporal_target": _build_temporal_target(
            sample,
            candidate_rows,
            frame_width=frame_width,
            frame_height=frame_height,
            action_window_ms=action_window_ms,
            settings=settings,
        ),
        "candidates": candidate_rows,
        "slider_paths": slider_rows,
    }


def _build_temporal_target(
    sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    action_window_ms: float,
    settings: Settings | None = None,
) -> dict[str, Any]:
    timestamp_ms = _optional_float(sample.get("timestamp_ms")) or 0.0
    target = _select_temporal_object(
        sample.get("hit_objects") or (),
        timestamp_ms=timestamp_ms,
        action_window_ms=action_window_ms,
    )
    if target is None:
        return {
            "target_strategy": "beatmap_action_v1",
            "target_strategy_version": "beatmap-action-v2",
            "action": "no_op",
            "action_id": 0,
            "selected_candidate_id": None,
            "time_offset_ms": 0.0,
        }
    transform, transform_spec = transform_from_settings_or_sample(
        settings,
        sample,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    video_xy = transform.osu_to_video(target["x"], target["y"])
    candidate = _nearest_candidate(candidates, video_xy)
    return {
        "target_strategy": "beatmap_action_v1",
        "target_strategy_version": "beatmap-action-v2",
        "coordinate_transform_version": transform_spec.version,
        "action": target["action"],
        "action_id": target["action_id"],
        "selected_candidate_id": (
            None if candidate is None else candidate.get("candidate_id")
        ),
        "target_video_xy": [float(video_xy[0]), float(video_xy[1])],
        "target_osu_xy": [float(target["x"]), float(target["y"])],
        "time_offset_ms": float(target["time_offset_ms"]),
        "object_type": target["object_type"],
        "source_index": target["source_index"],
        "object_start_ms": target["start_ms"],
        "object_end_ms": target["end_ms"],
    }


def _select_temporal_object(
    objects: object,
    *,
    timestamp_ms: float,
    action_window_ms: float,
) -> dict[str, Any] | None:
    if not isinstance(objects, Sequence):
        return None
    targets = [
        target
        for item in objects
        if isinstance(item, Mapping)
        for target in (_temporal_target_for_object(
            item,
            timestamp_ms=timestamp_ms,
            action_window_ms=action_window_ms,
        ),)
        if target is not None
    ]
    if not targets:
        return None
    priority = {"press": 0, "release": 1, "hold": 2}
    return min(
        targets,
        key=lambda target: (
            priority[target["action"]],
            abs(float(target["time_offset_ms"])),
            int(target["source_index"] if target["source_index"] is not None else 0),
        ),
    )


def _temporal_target_for_object(
    item: Mapping[str, Any],
    *,
    timestamp_ms: float,
    action_window_ms: float,
) -> dict[str, Any] | None:
    start_ms = _optional_float(item.get("start_ms"))
    end_ms = _optional_float(item.get("end_ms"))
    if start_ms is None:
        return None
    if end_ms is None:
        end_ms = start_ms
    point = _object_osu_point(item)
    if point is None:
        return None
    kind = _object_kind(item)
    action = None
    boundary_ms = start_ms
    boundary_point = point
    repeat_boundaries = _repeat_boundaries(item, start_ms=start_ms, end_ms=end_ms)
    if abs(timestamp_ms - start_ms) <= action_window_ms:
        action = "press"
        boundary_ms = start_ms
    elif kind == "circle" and _is_release_frame(
        timestamp_ms,
        start_ms=start_ms,
        action_window_ms=action_window_ms,
    ):
        action = "release"
        boundary_ms = start_ms + _click_duration_ms(action_window_ms)
    elif kind == "slider" and repeat_boundaries:
        selected = min(
            repeat_boundaries,
            key=lambda item: (abs(timestamp_ms - item[0]), 0 if item[1] == "press" else 1),
        )
        if abs(timestamp_ms - selected[0]) <= action_window_ms:
            action = selected[1]
            boundary_ms = selected[0]
            boundary_point = selected[2]
    elif kind in {"slider", "spinner"} and abs(timestamp_ms - end_ms) <= action_window_ms:
        action = "release"
        boundary_ms = end_ms
        if kind == "slider":
            boundary_point = _slider_tail_point(item) or point
    elif kind in {"slider", "spinner"} and start_ms < timestamp_ms < end_ms:
        action = "hold"
        boundary_ms = timestamp_ms
    if action is None:
        return None
    return {
        "action": action,
        "action_id": {"press": 1, "hold": 2, "release": 3}[action],
        "time_offset_ms": timestamp_ms - boundary_ms,
        "object_type": kind,
        "source_index": item.get("source_index"),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "x": boundary_point[0],
        "y": boundary_point[1],
    }


def _click_duration_ms(action_window_ms: float) -> float:
    return max(action_window_ms, 1.0)


def _is_release_frame(
    timestamp_ms: float,
    *,
    start_ms: float,
    action_window_ms: float,
) -> bool:
    release_ms = start_ms + _click_duration_ms(action_window_ms)
    return abs(timestamp_ms - release_ms) <= action_window_ms


def _repeat_boundaries(
    item: Mapping[str, Any],
    *,
    start_ms: float,
    end_ms: float,
) -> tuple[tuple[float, str, tuple[float, float]], ...]:
    repeats_raw = _optional_float(item.get("repeats"))
    repeats = max(1, int(repeats_raw or 1))
    if repeats <= 1 or end_ms <= start_ms:
        return ()
    head = _object_osu_point(item)
    tail = _slider_tail_point(item)
    if head is None or tail is None:
        return ()
    span = end_ms - start_ms
    boundaries: list[tuple[float, str, tuple[float, float]]] = []
    for repeat_index in range(1, repeats):
        time_ms = start_ms + span * repeat_index / repeats
        point = tail if repeat_index % 2 == 1 else head
        boundaries.append((time_ms, "release", point))
        boundaries.append((time_ms, "press", point))
    return tuple(boundaries)


def _slider_tail_point(item: Mapping[str, Any]) -> tuple[float, float] | None:
    path = item.get("path")
    if isinstance(path, Sequence) and path:
        last = path[-1]
        if isinstance(last, Sequence) and len(last) >= 2:
            return float(last[0]), float(last[1])
    return _object_osu_point(item)


def _object_osu_point(item: Mapping[str, Any]) -> tuple[float, float] | None:
    if item.get("x") is not None and item.get("y") is not None:
        return float(item["x"]), float(item["y"])
    path = item.get("path")
    if isinstance(path, Sequence) and path:
        first = path[0]
        if isinstance(first, Sequence) and len(first) >= 2:
            return float(first[0]), float(first[1])
    kind = _object_kind(item)
    if kind == "spinner":
        return 256.0, 192.0
    return None


def _object_kind(item: Mapping[str, Any]) -> str:
    raw = str(item.get("type", "")).lower()
    if "slider" in raw:
        return "slider"
    if "spinner" in raw:
        return "spinner"
    return "circle"


def _nearest_candidate(
    candidates: Sequence[Mapping[str, Any]],
    point: tuple[float, float],
) -> Mapping[str, Any] | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: _point_distance(
            point,
            (
                _optional_float(candidate.get("x")) or 0.0,
                _optional_float(candidate.get("y")) or 0.0,
            ),
        ),
    )


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


def _apply_candidate_reviews(
    rows: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    enabled: bool,
    max_candidates: int,
) -> None:
    if not enabled or max_candidates <= 0:
        return
    reviewed = 0
    for row in rows:
        reasons = tuple(row.get("ambiguity_reasons") or ())
        if not reasons:
            continue
        row["ambiguity_review"] = {
            "reasons": reasons,
            "strategy": "bounded_metadata_review",
            "resolved": False,
            "frame_bounds": [0, 0, frame_width, frame_height],
        }
        row["ambiguous"] = True
        reviewed += 1
        if reviewed >= max_candidates:
            break


def _apply_local_refinement(
    rows: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    enabled: bool,
    top_k: int,
    radius_px: float,
) -> None:
    if not enabled or top_k <= 0 or radius_px <= 0:
        return
    ordered = sorted(
        rows,
        key=lambda row: (
            not bool(row.get("ambiguous")),
            -float(row.get("score") or 0.0),
        ),
    )
    for row in ordered[:top_k]:
        before = [float(row.get("x") or 0.0), float(row.get("y") or 0.0)]
        after = [
            min(max(before[0], 0.0), float(frame_width - 1)),
            min(max(before[1], 0.0), float(frame_height - 1)),
        ]
        score = float(row.get("score") or 0.0)
        refined_score = min(1.0, score + 0.01 if row.get("ambiguous") else score)
        row["local_refinement"] = {
            "strategy": "bounded_topk_stride1_review",
            "triggered": True,
            "max_radius_px": radius_px,
            "before_xy": before,
            "after_xy": after,
            "before_score": score,
            "after_score": refined_score,
            "changed": before != after or refined_score != score,
        }
        row["x"], row["y"], row["score"] = after[0], after[1], refined_score


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


def _optional_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


__all__ = [
    "CANDIDATE_CACHE_VERSION",
    "CandidateCacheBuildResult",
    "build_candidate_cache_record",
    "generate_candidate_cache",
]
