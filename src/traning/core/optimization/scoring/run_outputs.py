"""对齐候选缓存与决策输出，并统一换算坐标后进行序列评分。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from package.coordinates import frame_normalized_to_pixel
from traning.lib.coordinates import transform_from_settings_or_sample
from traning.lib.metrics import PredictedClick, TargetObject
from traning.lib.training import DEFAULT_CIRCLE_RADIUS_OSU_PIXELS
from traning.state import TrialParameters
from traning.core.optimization.scoring.evaluator import (
    SampleScoringInput,
    TrialScoreReport,
    TrialScoreSpec,
    score_trial,
)


# 保留旧公开名称；唯一默认值由训练标签模块定义，避免各阶段再次漂移。
DEFAULT_CIRCLE_RADIUS_OSU = DEFAULT_CIRCLE_RADIUS_OSU_PIXELS


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
    circle_radius: float | None = None,
    spec: TrialScoreSpec = TrialScoreSpec(),
    settings: Any | None = None,
) -> DecisionOutputScoreResult:
    if not parameter_group_id:
        raise ValueError("parameter_group_id must not be empty")
    candidate_rows = tuple(_read_jsonl(candidate_cache_path))
    decision_rows = tuple(_read_jsonl(decisions_path))
    if not candidate_rows:
        raise ValueError("candidate cache must contain at least one frame")
    cache_by_key = _index_unique_frames(candidate_rows, label="candidate cache")
    decisions_by_key = _index_unique_frames(decision_rows, label="decision output")
    missing_decisions = tuple(sorted(cache_by_key.keys() - decisions_by_key.keys()))
    orphan_decisions = tuple(sorted(decisions_by_key.keys() - cache_by_key.keys()))
    if missing_decisions or orphan_decisions:
        raise ValueError(
            "candidate/decision frame keys do not match: "
            f"missing_decisions={len(missing_decisions)} "
            f"orphan_decisions={len(orphan_decisions)} "
            f"missing_preview={missing_decisions[:3]} "
            f"orphan_preview={orphan_decisions[:3]}"
        )
    samples = []
    no_op_frames = 0
    action_frames = 0
    # 以候选缓存作为固定评估集顺序；上方已要求两侧 key
    # 一一对应，因此决策漏帧不会被严格 sample gate 静默忽略。
    for cache_row in candidate_rows:
        decision = decisions_by_key[_frame_key(cache_row)]
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
                circle_radius_override=circle_radius,
                settings=settings,
            )
        )
    report = score_trial(
        parameter_group_id,
        samples,
        parameters=TrialParameters(training={"parameter_group_id": parameter_group_id}),
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
    circle_radius_override: float | None,
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
    temporal_target = cache_row.get("temporal_target")
    target_metadata = temporal_target if isinstance(temporal_target, Mapping) else {}
    circle_radius = _circle_radius_from_row(
        cache_row,
        target_metadata,
        override=circle_radius_override,
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
            "candidate_count": len(cache_row.get("candidates") or ()),
            "circle_radius_osu_pixels": circle_radius,
            "candidate_match_status": target_metadata.get("candidate_match_status"),
            "candidate_match_unmatched_reason": target_metadata.get(
                "candidate_match_unmatched_reason"
            ),
            "transform_status": (
                (cache_row.get("coordinate_transform") or {}).get("transform_status")
                if isinstance(cache_row.get("coordinate_transform"), Mapping)
                else None
            ),
        },
    )


def _circle_radius_from_row(
    cache_row: Mapping[str, Any],
    target_metadata: Mapping[str, Any],
    *,
    override: float | None,
) -> float:
    """按显式覆盖、新缓存字段、目标字段、旧协议默认值依次解析半径。"""

    candidates = (
        override,
        _safe_float(cache_row.get("circle_radius_osu_pixels")),
        _safe_float(target_metadata.get("candidate_match_radius_osu")),
        DEFAULT_CIRCLE_RADIUS_OSU,
    )
    for value in candidates:
        if value is not None and value > 0:
            return float(value)
    return DEFAULT_CIRCLE_RADIUS_OSU


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
    start_ms = timestamp_ms - (_safe_float(target.get("time_offset_ms")) or 0.0)
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
    """把决策位置统一转换为 osu! 坐标后构造评分点击事件。"""
    if str(decision.get("action") or "no_op") == "no_op":
        return ()
    timestamp_ms = _safe_float(decision.get("timestamp_ms"))
    if timestamp_ms is None:
        timestamp_ms = _safe_float(cache_row.get("timestamp_ms")) or 0.0
    time_ms = timestamp_ms - (_safe_float(decision.get("time_offset_ms")) or 0.0)
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
        # 没有可用候选像素时，模型输出仍是整帧归一化坐标，必须依次执行
        # frame normalized -> frame pixel -> osu，而不能直接乘 512x384。
        point = _normalized_frame_to_osu(
            decision.get("predicted_xy_normalized"),
            cache_row,
            settings=settings,
        )
    if point is None:
        return ()
    return (PredictedClick(time_ms=time_ms, x=point[0], y=point[1]),)


def _prediction_video_xy(
    cache_row: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[float, float] | None:
    """解析决策对应的视频像素，优先使用被选候选的原始像素位置。

    候选缓存中的 x/y 是检测阶段保存的视频像素，比从归一化特征反算更直接；
    仅在候选缺失或无法匹配时才回退到模型回归的整帧归一化坐标。
    """
    if str(decision.get("action") or "no_op") == "no_op":
        return None
    selected_id = _safe_int(decision.get("selected_candidate_id"))
    if selected_id is not None:
        # selected_candidate_id 是跨阶段的稳定引用；从缓存回查原始像素，
        # 不使用 decision 中仅供诊断的 normalized candidate feature。
        candidates = cache_row.get("candidates")
        if isinstance(candidates, Sequence) and not isinstance(
            candidates,
            (str, bytes),
        ):
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                if _safe_int(candidate.get("candidate_id")) != selected_id:
                    continue
                x = _safe_float(candidate.get("x"))
                y = _safe_float(candidate.get("y"))
                if x is not None and y is not None:
                    return (x, y)
                break
    # 回归坐标的归一化基准是完整视频帧，而不是 osu! playfield。
    normalized = _point_pair(decision.get("predicted_xy_normalized"))
    frame_width = _safe_int(cache_row.get("frame_width"))
    frame_height = _safe_int(cache_row.get("frame_height"))
    if normalized is None or frame_width is None or frame_height is None:
        return None
    return frame_normalized_to_pixel(
        *normalized,
        width=frame_width,
        height=frame_height,
    )


def _video_to_osu_pair(
    value: object,
    row: Mapping[str, Any],
    *,
    settings: Any | None = None,
) -> tuple[float, float] | None:
    """校验一个视频像素坐标对，并使用样本变换映射到 osu! 空间。"""
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
    """使用与当前缓存样本匹配的变换执行 frame pixel -> osu。"""
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


def _normalized_frame_to_osu(
    value: object,
    row: Mapping[str, Any],
    *,
    settings: Any | None = None,
) -> tuple[float, float] | None:
    """按整帧尺寸还原模型坐标，再由样本变换转换到 osu! 空间。"""
    point = _point_pair(value)
    if point is None:
        return None
    frame_width = _safe_int(row.get("frame_width"))
    frame_height = _safe_int(row.get("frame_height"))
    if frame_width is None or frame_height is None:
        return None
    # 两步转换显式分开，保证 affine/explicit_rect/legacy 等变换均从同一
    # 视频像素空间接收输入，并与训练标签采用的坐标契约一致。
    video_xy = frame_normalized_to_pixel(
        point[0],
        point[1],
        width=frame_width,
        height=frame_height,
    )
    return _video_to_osu(*video_xy, row, settings=settings)


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


def _index_unique_frames(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    """校验帧身份并构建唯一索引，避免字典覆盖重复帧。"""

    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        key = _frame_key(row)
        if not key[0] or key[1] < 0:
            raise ValueError(f"{label} row {row_number} has invalid frame key: {key}")
        if key in indexed:
            raise ValueError(f"{label} contains duplicate frame key: {key}")
        indexed[key] = row
    return indexed


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
