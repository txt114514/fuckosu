from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from package.coordinates import (
    OSU_PLAYFIELD_HEIGHT,
    OSU_PLAYFIELD_WIDTH,
)
from traning.lib.coordinates import transform_from_settings_or_sample
from traning.lib.metrics import PredictedClick, TargetObject
from traning.state import TrialParameters
from traning.core.optimization.scoring.evaluator import (
    SampleScoringInput,
    TrialScoreReport,
    TrialScoreSpec,
    score_trial,
)


DEFAULT_CIRCLE_RADIUS_OSU = 32.0


@dataclass(frozen=True)
class DecisionOutputScoreResult:
    parameter_group_id: str
    report: TrialScoreReport
    candidate_frame_count: int
    decision_frame_count: int
    no_op_frame_count: int
    action_frame_count: int

    def as_summary(self) -> dict[str, Any]:
        return {
            "parameter_group_id": self.parameter_group_id,
            "quality_score": self.report.quality_score,
            "passed": self.report.passed,
            "samples": len(self.report.samples),
            "candidate_frames": self.candidate_frame_count,
            "decision_frames": self.decision_frame_count,
            "no_op_frames": self.no_op_frame_count,
            "action_frames": self.action_frame_count,
            "targets": self.report.target_count,
            "hits": self.report.hit_count,
            "misses": self.report.miss_count,
            "unresolved": self.report.unresolved_count,
            "frequency_limited": self.report.frequency_limited_count,
        }


def score_decision_outputs(
    *,
    parameter_group_id: str,
    candidate_cache_path: Path,
    decisions_path: Path,
    metrics: Mapping[str, float] | None = None,
    circle_radius: float = DEFAULT_CIRCLE_RADIUS_OSU,
    spec: TrialScoreSpec = TrialScoreSpec(),
    settings: Any | None = None,
) -> DecisionOutputScoreResult:
    if not parameter_group_id:
        raise ValueError("parameter_group_id must not be empty")
    candidate_rows = tuple(_read_jsonl(candidate_cache_path))
    decision_rows = tuple(_read_jsonl(decisions_path))
    cache_by_key = {
        _frame_key(row): row
        for row in candidate_rows
    }
    samples = []
    no_op_frames = 0
    action_frames = 0
    for decision in decision_rows:
        key = _frame_key(decision)
        cache_row = cache_by_key.get(key)
        if cache_row is None:
            continue
        action = str(decision.get("action") or "no_op")
        if action == "no_op":
            no_op_frames += 1
        else:
            action_frames += 1
        samples.append(
            _sample_from_rows(
                cache_row,
                decision,
                parameter_group_id=parameter_group_id,
                circle_radius=circle_radius,
                settings=settings,
            )
        )
    report = score_trial(
        parameter_group_id,
        samples,
        parameters=TrialParameters(
            training={"parameter_group_id": parameter_group_id}
        ),
        metrics={
            "candidate_frame_count": float(len(candidate_rows)),
            "decision_frame_count": float(len(decision_rows)),
            "scored_frame_count": float(len(samples)),
            "no_op_frame_count": float(no_op_frames),
            "action_frame_count": float(action_frames),
            **dict(metrics or {}),
        },
        spec=spec,
    )
    return DecisionOutputScoreResult(
        parameter_group_id=parameter_group_id,
        report=report,
        candidate_frame_count=len(candidate_rows),
        decision_frame_count=len(decision_rows),
        no_op_frame_count=no_op_frames,
        action_frame_count=action_frames,
    )


def _sample_from_rows(
    cache_row: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    parameter_group_id: str,
    circle_radius: float,
    settings: Any | None = None,
) -> SampleScoringInput:
    sample_key = str(cache_row.get("sample_key") or decision.get("sample_key"))
    frame_index = _safe_int(cache_row.get("frame_index")) or 0
    targets = _target_objects(cache_row, settings=settings)
    predicted_video_xy = _prediction_video_xy(cache_row, decision)
    predictions = _predicted_clicks(
        cache_row,
        decision,
        predicted_video_xy=predicted_video_xy,
        settings=settings,
    )
    return SampleScoringInput(
        sample_key=sample_key,
        subproject=_subproject_from_sample_key(sample_key),
        targets=targets,
        predictions=predictions,
        circle_radius=circle_radius,
        frame_index=frame_index,
        metadata={
            "parameter_group_id": parameter_group_id,
            "action": decision.get("action"),
            "action_id": decision.get("action_id"),
            "action_probability": decision.get("action_probability"),
            "selected_candidate_id": decision.get("selected_candidate_id"),
            "selected_candidate_probability": (
                decision.get("selected_candidate_probability")
            ),
            "predicted_video_xy": predicted_video_xy,
            "time_offset_ms": decision.get("time_offset_ms"),
        },
    )


def _target_objects(
    row: Mapping[str, Any],
    *,
    settings: Any | None = None,
) -> tuple[TargetObject, ...]:
    target = row.get("temporal_target")
    if not isinstance(target, Mapping):
        return ()
    if str(target.get("action") or "no_op") == "no_op":
        return ()
    target_xy = _point_pair(target.get("target_osu_xy"))
    if target_xy is None:
        target_xy = _video_to_osu_pair(
            target.get("target_video_xy"),
            row,
            settings=settings,
        )
    if target_xy is None:
        return ()
    timestamp_ms = _safe_float(row.get("timestamp_ms")) or 0.0
    start_ms = _safe_float(target.get("object_start_ms"))
    if start_ms is None:
        start_ms = timestamp_ms + (_safe_float(target.get("time_offset_ms")) or 0.0)
    end_ms = _safe_float(target.get("object_end_ms"))
    if end_ms is None or end_ms < start_ms:
        end_ms = start_ms
    source_index = _safe_int(target.get("source_index"))
    return (
        TargetObject(
            target_id=(
                f"{row.get('sample_key')}:{row.get('frame_index')}:"
                f"{source_index if source_index is not None else 'target'}"
            ),
            target_type="circle",
            start_ms=start_ms,
            end_ms=end_ms,
            x=target_xy[0],
            y=target_xy[1],
            source_index=source_index,
        ),
    )


def _predicted_clicks(
    cache_row: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    predicted_video_xy: tuple[float, float] | None = None,
    settings: Any | None = None,
) -> tuple[PredictedClick, ...]:
    if str(decision.get("action") or "no_op") == "no_op":
        return ()
    timestamp_ms = _safe_float(decision.get("timestamp_ms"))
    if timestamp_ms is None:
        timestamp_ms = _safe_float(cache_row.get("timestamp_ms")) or 0.0
    time_ms = timestamp_ms + (_safe_float(decision.get("time_offset_ms")) or 0.0)
    point = (
        None
        if predicted_video_xy is None
        else _video_to_osu(
            predicted_video_xy[0],
            predicted_video_xy[1],
            cache_row,
            settings=settings,
        )
    )
    if point is None:
        point = _normalized_to_osu(decision.get("predicted_xy_normalized"))
    if point is None:
        return ()
    return (PredictedClick(time_ms=time_ms, x=point[0], y=point[1]),)


def _prediction_video_xy(
    cache_row: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[float, float] | None:
    if str(decision.get("action") or "no_op") == "no_op":
        return None
    selected_id = _safe_int(decision.get("selected_candidate_id"))
    if selected_id is None:
        return None
    candidates = cache_row.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if _safe_int(candidate.get("candidate_id")) != selected_id:
            continue
        x = _safe_float(candidate.get("x"))
        y = _safe_float(candidate.get("y"))
        if x is None or y is None:
            return None
        return (x, y)
    return None


def _video_to_osu_pair(
    value: object,
    row: Mapping[str, Any],
    *,
    settings: Any | None = None,
) -> tuple[float, float] | None:
    point = _point_pair(value)
    if point is None:
        return None
    return _video_to_osu(point[0], point[1], row, settings=settings)


def _video_to_osu(
    x: float,
    y: float,
    row: Mapping[str, Any],
    *,
    settings: Any | None = None,
) -> tuple[float, float] | None:
    frame_width = _safe_int(row.get("frame_width"))
    frame_height = _safe_int(row.get("frame_height"))
    if frame_width is None or frame_height is None:
        return None
    transform, _ = transform_from_settings_or_sample(
        settings,
        row,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return transform.video_to_osu(x, y)


def _normalized_to_osu(value: object) -> tuple[float, float] | None:
    point = _point_pair(value)
    if point is None:
        return None
    return (point[0] * OSU_PLAYFIELD_WIDTH, point[1] * OSU_PLAYFIELD_HEIGHT)


def _point_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    x = _safe_float(value[0])
    y = _safe_float(value[1])
    if x is None or y is None:
        return None
    return (x, y)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _frame_key(row: Mapping[str, Any]) -> tuple[str, int]:
    sample_key = str(row.get("sample_key") or "")
    frame_index = _safe_int(row.get("frame_index"))
    return sample_key, frame_index if frame_index is not None else -1


def _subproject_from_sample_key(sample_key: str) -> str:
    if "long_sequence" in sample_key:
        return "long_sequence"
    if "dual_point" in sample_key:
        return "dual_point"
    return "single_point"


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_CIRCLE_RADIUS_OSU",
    "DecisionOutputScoreResult",
    "score_decision_outputs",
]
