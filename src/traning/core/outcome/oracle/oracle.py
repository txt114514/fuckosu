"""仅供离线训练构造反事实 Outcome 标签的 canonical oracle。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from traning.state import (
    OUTCOME_LOW_SCORE_UPPER,
    OUTCOME_MEDIUM_SCORE_UPPER,
    ObjectType,
    OutcomeCategory,
    Point2D,
)
from traning.core.evaluation import (
    PredictedClick,
    ScoreSpec,
    SequenceScore,
    SequenceScoreSpec,
    TargetObject,
    score_click_sequence,
    score_point,
    score_slider,
)


OUTCOME_ORACLE_VERSION = "outcome-oracle-v1"


@dataclass(frozen=True, slots=True)
class OracleTarget:
    """离线 oracle 可见的单个目标真值，不进入 runtime。"""

    track_id: str
    object_id: str
    object_type: ObjectType
    position: Point2D
    start_time_ms: float
    end_time_ms: float
    path: tuple[Point2D, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.track_id, "track_id")
        _require_identifier(self.object_id, "object_id")
        if not isinstance(self.object_type, ObjectType):
            raise TypeError("object_type 必须是 ObjectType")
        if not isinstance(self.position, Point2D):
            raise TypeError("position 必须是 Point2D")
        _require_nonnegative(self.start_time_ms, "start_time_ms")
        _require_nonnegative(self.end_time_ms, "end_time_ms")
        if self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms 不得早于 start_time_ms")
        _require_point_path(self.path, "path")
        if self.object_type is ObjectType.RING and self.path:
            raise ValueError("ring target 只能使用 position，不得携带 path")
        if self.object_type is ObjectType.SLIDER and len(self.path) < 2:
            raise ValueError("slider target 的 path 至少需要两个点")
        if self.object_type is ObjectType.SLIDER and self.position != self.path[0]:
            raise ValueError("slider target.position 必须等于 path 起点")
        if self.object_type in (ObjectType.SPINNER, ObjectType.UNKNOWN) and self.path:
            raise ValueError("spinner/unknown target 不得携带 path")


@dataclass(frozen=True, slots=True)
class OracleState:
    """某一时刻离线 oracle 的不可变目标快照。"""

    state_id: str
    timestamp_ms: float
    targets: tuple[OracleTarget, ...]
    resolved_track_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.state_id, "state_id")
        _require_nonnegative(self.timestamp_ms, "timestamp_ms")
        if not isinstance(self.targets, tuple) or any(
            not isinstance(target, OracleTarget) for target in self.targets
        ):
            raise TypeError("targets 必须是 OracleTarget 元组")
        track_ids = tuple(target.track_id for target in self.targets)
        object_ids = tuple(target.object_id for target in self.targets)
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("targets 的 track_id 必须唯一")
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("targets 的 object_id 必须唯一")
        if not isinstance(self.resolved_track_ids, tuple):
            raise TypeError("resolved_track_ids 必须是字符串元组")
        for track_id in self.resolved_track_ids:
            _require_identifier(track_id, "resolved_track_ids[]")
        if len(self.resolved_track_ids) != len(set(self.resolved_track_ids)):
            raise ValueError("resolved_track_ids 不得重复")
        if not set(self.resolved_track_ids).issubset(track_ids):
            raise ValueError("resolved_track_ids 必须引用 targets 中的 track_id")


@dataclass(frozen=True, slots=True)
class HypotheticalClick:
    """对指定轨迹和未来 horizon 的离线反事实点击。"""

    track_id: str
    horizon_ms: float
    position: Point2D
    path: tuple[Point2D, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.track_id, "track_id")
        _require_nonnegative(self.horizon_ms, "horizon_ms")
        if not isinstance(self.position, Point2D):
            raise TypeError("position 必须是 Point2D")
        _require_point_path(self.path, "path")
        if len(self.path) == 1:
            raise ValueError("hypothetical path 必须为空或至少包含两个点")


@dataclass(frozen=True, slots=True)
class OracleOutcome:
    """离线 canonical score 到五分类 Outcome 标签的完整投影。"""

    track_id: str
    horizon_ms: float
    category: OutcomeCategory
    score: float
    valid: bool
    expires: bool
    passed: bool
    target_object_id: str | None
    spatial_error: float | None
    time_error_ms: float | None

    def __post_init__(self) -> None:
        _require_identifier(self.track_id, "track_id")
        _require_nonnegative(self.horizon_ms, "horizon_ms")
        if not isinstance(self.category, OutcomeCategory):
            raise TypeError("category 必须是 OutcomeCategory")
        _require_probability(self.score, "score")
        for field_name, value in (
            ("valid", self.valid),
            ("expires", self.expires),
            ("passed", self.passed),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} 必须是 bool")
        if self.target_object_id is not None:
            _require_identifier(self.target_object_id, "target_object_id")
        for field_name, value in (
            ("spatial_error", self.spatial_error),
            ("time_error_ms", self.time_error_ms),
        ):
            if value is not None:
                _require_nonnegative(value, field_name)
        if self.category is OutcomeCategory.INVALID:
            if self.valid or self.passed or self.score != 0.0:
                raise ValueError(
                    "INVALID outcome 必须 valid=False、passed=False、score=0"
                )
            if self.spatial_error is not None or self.time_error_ms is not None:
                raise ValueError("INVALID outcome 不得携带评分误差")
        else:
            if not self.valid or self.expires or self.target_object_id is None:
                raise ValueError("非 INVALID outcome 必须有效、未过期并引用目标")
            expected_passed = self.category in (
                OutcomeCategory.LOW,
                OutcomeCategory.MEDIUM,
                OutcomeCategory.HIGH,
            )
            if self.passed is not expected_passed:
                raise ValueError("passed 必须与 OutcomeCategory 一致")
            if self.spatial_error is None or self.time_error_ms is None:
                raise ValueError("有效 outcome 必须携带完整评分误差")
        if self.expires and self.category is not OutcomeCategory.INVALID:
            raise ValueError("expires=True 仅适用于 INVALID outcome")


@dataclass(frozen=True, slots=True)
class OutcomeOracle:
    """调用唯一 V2 评分实现生成离线 Outcome 监督。"""

    circle_radius: float
    spec: ScoreSpec = ScoreSpec()

    def __post_init__(self) -> None:
        _require_positive(self.circle_radius, "circle_radius")
        if not isinstance(self.spec, ScoreSpec):
            raise TypeError("spec 必须是 ScoreSpec")

    def evaluate(
        self, state: OracleState, hypothetical_action: HypotheticalClick
    ) -> OracleOutcome:
        """在 ``state.timestamp + horizon`` 执行反事实点击并生成标签。"""

        if not isinstance(state, OracleState):
            raise TypeError("state 必须是 OracleState")
        if not isinstance(hypothetical_action, HypotheticalClick):
            raise TypeError("hypothetical_action 必须是 HypotheticalClick")
        action = hypothetical_action
        target_by_track = {target.track_id: target for target in state.targets}
        target = target_by_track.get(action.track_id)
        if target is None:
            return _invalid_outcome(action, target_object_id=None, expires=False)

        execution_time_ms = state.timestamp_ms + action.horizon_ms
        expires = (
            execution_time_ms > target.end_time_ms + self.spec.temporal_pass_end_ms
        )
        if expires:
            return _invalid_outcome(
                action, target_object_id=target.object_id, expires=True
            )
        if action.track_id in state.resolved_track_ids:
            return _invalid_outcome(
                action, target_object_id=target.object_id, expires=False
            )
        if target.object_type not in (ObjectType.RING, ObjectType.SLIDER):
            return _invalid_outcome(
                action, target_object_id=target.object_id, expires=False
            )

        reference_position = _point_tuple(target.position)
        predicted_position = _point_tuple(action.position)
        if target.object_type is ObjectType.RING:
            if action.path:
                return _invalid_outcome(
                    action, target_object_id=target.object_id, expires=False
                )
            return self._evaluate_point(
                action,
                target=target,
                execution_time_ms=execution_time_ms,
                reference_position=reference_position,
                predicted_position=predicted_position,
            )

        # 空 path 表示仅评估 slider head 点击；完整路径仅在离线可得时参与评分。
        if not action.path:
            return self._evaluate_point(
                action,
                target=target,
                execution_time_ms=execution_time_ms,
                reference_position=reference_position,
                predicted_position=predicted_position,
            )
        slider_score = score_slider(
            reference_head_xy=reference_position,
            predicted_head_xy=predicted_position,
            reference_path=tuple(_point_tuple(point) for point in target.path),
            predicted_path=tuple(_point_tuple(point) for point in action.path),
            circle_radius=self.circle_radius,
            reference_start_ms=target.start_time_ms,
            predicted_start_ms=execution_time_ms,
            spec=self.spec,
        )
        return _scored_outcome(
            action,
            target=target,
            normalized_score=slider_score.score.normalized,
            passed=slider_score.passed,
            spatial_error=slider_score.head.distance,
            time_error_ms=slider_score.head.time_error_ms,
        )

    def evaluate_sequence(
        self,
        targets: tuple[TargetObject, ...],
        clicks: tuple[PredictedClick, ...],
        *,
        min_click_interval_ms: float = 50.0,
    ) -> SequenceScore:
        """委托 canonical sequence scorer 完成匹配、频率限制和错误归因。"""

        sequence_spec = SequenceScoreSpec(
            min_click_interval_ms=min_click_interval_ms,
            object_score_spec=self.spec,
        )
        return score_click_sequence(
            targets,
            clicks,
            circle_radius=self.circle_radius,
            spec=sequence_spec,
        )

    def _evaluate_point(
        self,
        action: HypotheticalClick,
        *,
        target: OracleTarget,
        execution_time_ms: float,
        reference_position: tuple[float, float],
        predicted_position: tuple[float, float],
    ) -> OracleOutcome:
        """使用 canonical point score 评估 ring 或 slider head。"""

        point_score = score_point(
            reference_position,
            predicted_position,
            circle_radius=self.circle_radius,
            reference_time_ms=target.start_time_ms,
            predicted_time_ms=execution_time_ms,
            spec=self.spec,
        )
        return _scored_outcome(
            action,
            target=target,
            normalized_score=point_score.score.normalized,
            passed=point_score.passed,
            spatial_error=point_score.distance,
            time_error_ms=point_score.time_error_ms,
        )


def _invalid_outcome(
    action: HypotheticalClick, *, target_object_id: str | None, expires: bool
) -> OracleOutcome:
    return OracleOutcome(
        track_id=action.track_id,
        horizon_ms=action.horizon_ms,
        category=OutcomeCategory.INVALID,
        score=0.0,
        valid=False,
        expires=expires,
        passed=False,
        target_object_id=target_object_id,
        spatial_error=None,
        time_error_ms=None,
    )


def _scored_outcome(
    action: HypotheticalClick,
    *,
    target: OracleTarget,
    normalized_score: float,
    passed: bool,
    spatial_error: float,
    time_error_ms: float,
) -> OracleOutcome:
    _require_probability(normalized_score, "canonical normalized score")
    if not passed:
        category = OutcomeCategory.MISS
    elif normalized_score < OUTCOME_LOW_SCORE_UPPER:
        category = OutcomeCategory.LOW
    elif normalized_score < OUTCOME_MEDIUM_SCORE_UPPER:
        category = OutcomeCategory.MEDIUM
    else:
        category = OutcomeCategory.HIGH
    return OracleOutcome(
        track_id=action.track_id,
        horizon_ms=action.horizon_ms,
        category=category,
        score=float(normalized_score),
        valid=True,
        expires=False,
        passed=passed,
        target_object_id=target.object_id,
        spatial_error=spatial_error,
        time_error_ms=time_error_ms,
    )


def _point_tuple(point: Point2D) -> tuple[float, float]:
    return point.x, point.y


def _require_point_path(path: tuple[Point2D, ...], field_name: str) -> None:
    if not isinstance(path, tuple) or any(
        not isinstance(point, Point2D) for point in path
    ):
        raise TypeError(f"{field_name} 必须是 Point2D 元组")


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} 必须非空且无首尾空格")


def _require_nonnegative(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} 必须是数值")
    if not math.isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{field_name} 必须是有限非负数")


def _require_positive(value: float, field_name: str) -> None:
    _require_nonnegative(value, field_name)
    if value <= 0.0:
        raise ValueError(f"{field_name} 必须大于 0")


def _require_probability(value: float, field_name: str) -> None:
    _require_nonnegative(value, field_name)
    if value > 1.0:
        raise ValueError(f"{field_name} 必须位于 [0, 1]")


__all__ = (
    "OUTCOME_ORACLE_VERSION",
    "HypotheticalClick",
    "OracleOutcome",
    "OracleState",
    "OracleTarget",
    "OutcomeCategory",
    "OutcomeOracle",
)
