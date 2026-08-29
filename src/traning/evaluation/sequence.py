"""按时间稳定匹配点击与目标，并归因空间、时间或决策错误。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, TypeAlias

from traning.data.coordinates import (
    FrameCoordinateTransform,
    FramePixelPoint,
    OsuPoint,
)

from .scoring import (
    PathPoints,
    PointScore,
    ScoreSpec,
    SliderScore,
    score_point,
    score_slider,
)


TargetType: TypeAlias = Literal["circle", "slider"]
ClickStatus: TypeAlias = Literal["hit", "miss", "frequency_limited"]
ErrorDomain: TypeAlias = Literal["none", "spatial", "temporal", "decision"]
ErrorTag: TypeAlias = Literal[
    "better_score_after_resolution",
    "duplicate_after_hit",
    "early_click",
    "frequency_limited",
    "head_spatial_miss",
    "late_click",
    "no_active_target",
    "slider_path_miss",
    "spatial_miss",
]


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是数值")
    if not isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")


def _valid_path(name: str, path: PathPoints) -> None:
    if not isinstance(path, tuple):
        raise TypeError(f"{name} 必须是坐标 tuple")
    for point_index, point in enumerate(path):
        if not isinstance(point, tuple) or len(point) != 2:
            raise TypeError(f"{name}[{point_index}] 必须是二元坐标 tuple")
        for value_index, value in enumerate(point):
            _finite(f"{name}[{point_index}][{value_index}]", value)


@dataclass(frozen=True, slots=True)
class SequenceScoreSpec:
    """序列级点击频率限制与单物件评分规格。"""

    min_click_interval_ms: float = 50.0
    object_score_spec: ScoreSpec = field(default_factory=ScoreSpec)

    def __post_init__(self) -> None:
        _finite("min_click_interval_ms", self.min_click_interval_ms)
        if self.min_click_interval_ms < 0:
            raise ValueError("min_click_interval_ms 不得为负数")
        if not isinstance(self.object_score_spec, ScoreSpec):
            raise TypeError("object_score_spec 必须是 ScoreSpec")


@dataclass(frozen=True, slots=True)
class TargetObject:
    """序列 oracle 的 canonical circle/slider 目标。"""

    target_id: str
    target_type: TargetType
    start_ms: float
    end_ms: float
    x: float | None = None
    y: float | None = None
    path: PathPoints = ()
    source_index: int | None = None
    frame_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str):
            raise TypeError("target_id 必须是字符串")
        if not self.target_id or self.target_id != self.target_id.strip():
            raise ValueError("target_id 不得为空且不得有首尾空格")
        if self.target_type not in ("circle", "slider"):
            raise ValueError("target_type 必须是 circle 或 slider")
        _finite("start_ms", self.start_ms)
        _finite("end_ms", self.end_ms)
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms 不得早于 start_ms")
        if self.x is not None:
            _finite("x", self.x)
        if self.y is not None:
            _finite("y", self.y)
        _valid_path("path", self.path)
        if self.target_type == "circle" and (self.x is None or self.y is None):
            raise ValueError("circle 目标必须提供 x 和 y")
        if self.target_type == "slider" and not self.path:
            raise ValueError("slider 目标必须提供 path")
        if self.source_index is not None:
            if isinstance(self.source_index, bool) or not isinstance(
                self.source_index, int
            ):
                raise TypeError("source_index 必须是整数或 None")
        if self.frame_index is not None and (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index 必须是非负整数或 None")


@dataclass(frozen=True, slots=True)
class PredictedClick:
    """按时间发布的 canonical osu! 预测点击与可选 slider 路径。"""

    time_ms: float
    x: float
    y: float
    path: PathPoints = ()

    def __post_init__(self) -> None:
        _finite("time_ms", self.time_ms)
        _finite("x", self.x)
        _finite("y", self.y)
        _valid_path("path", self.path)


@dataclass(frozen=True, slots=True)
class FramePredictedClick:
    """正式 evaluation 边界接收的原帧像素点击及可选 slider 路径。"""

    time_ms: float
    position: FramePixelPoint
    path: tuple[FramePixelPoint, ...] = ()
    frame_index: int | None = None

    def __post_init__(self) -> None:
        """确保路径点不能逃离帧尺寸与坐标指纹领域对象。"""

        _finite("time_ms", self.time_ms)
        if not isinstance(self.position, FramePixelPoint):
            raise TypeError("position 必须是 FramePixelPoint")
        if not isinstance(self.path, tuple) or any(
            not isinstance(point, FramePixelPoint) for point in self.path
        ):
            raise TypeError("path 必须是 FramePixelPoint 元组")
        if len(self.path) == 1:
            raise ValueError("slider path 必须为空或至少包含两个点")
        if self.frame_index is not None and (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index 必须是非负整数或 None")
        if any(
            point.source_frame_width != self.position.source_frame_width
            or point.source_frame_height != self.position.source_frame_height
            or point.transform_fingerprint != self.position.transform_fingerprint
            for point in self.path
        ):
            raise ValueError("position 与 path 必须共享原帧尺寸和坐标变换指纹")


@dataclass(frozen=True, slots=True)
class TargetResolution:
    """目标首次不可逆命中的解析记录。"""

    target_id: str
    source_index: int | None
    click_index: int
    click_time_ms: float
    score: PointScore | SliderScore


@dataclass(frozen=True, slots=True)
class ClickEvaluation:
    """单次点击的状态、评分和错误归因。"""

    click_index: int
    click: PredictedClick
    status: ClickStatus
    target_id: str | None = None
    source_index: int | None = None
    score: PointScore | SliderScore | None = None
    primary_error: ErrorDomain = "none"
    error_tags: tuple[ErrorTag, ...] = ()
    spatial_error: float | None = None
    temporal_error_ms: float | None = None

    @property
    def frequency_limited(self) -> bool:
        """说明本次点击是否仅因频率限制而未参与匹配。"""

        return self.status == "frequency_limited"


@dataclass(frozen=True, slots=True)
class SequenceScore:
    """完整点击序列的评分与未解析目标。"""

    clicks: tuple[ClickEvaluation, ...]
    resolved_targets: tuple[TargetResolution, ...]
    unresolved_target_ids: tuple[str, ...]

    @property
    def hit_count(self) -> int:
        """返回已被首次不可逆解析的目标数量。"""

        return len(self.resolved_targets)

    @property
    def miss_count(self) -> int:
        """返回 canonical 状态为普通 miss 的点击数量。"""

        return sum(item.status == "miss" for item in self.clicks)

    @property
    def frequency_limited_count(self) -> int:
        """返回因点击频率限制而跳过的点击数量。"""

        return sum(item.status == "frequency_limited" for item in self.clicks)


@dataclass(frozen=True, slots=True)
class FrameSequenceScore:
    """保留原帧点击、尺寸和变换指纹的 sequence score 信封。"""

    result: SequenceScore
    frame_clicks: tuple[FramePredictedClick, ...]
    source_frame_width: int
    source_frame_height: int
    transform_fingerprint: str
    unresolved_target_frame_indices: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.result, SequenceScore):
            raise TypeError("result 必须是 SequenceScore")
        if not isinstance(self.frame_clicks, tuple) or any(
            not isinstance(click, FramePredictedClick) for click in self.frame_clicks
        ):
            raise TypeError("frame_clicks 必须是 FramePredictedClick 元组")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.source_frame_width, self.source_frame_height)
        ):
            raise ValueError("source frame width/height 必须是正整数")
        if not isinstance(
            self.transform_fingerprint, str
        ) or not self.transform_fingerprint.startswith("transform-"):
            raise ValueError("transform_fingerprint 必须是有效坐标变换指纹")
        points = tuple(
            point
            for click in self.frame_clicks
            for point in (click.position, *click.path)
        )
        if any(
            point.source_frame_width != self.source_frame_width
            or point.source_frame_height != self.source_frame_height
            or point.transform_fingerprint != self.transform_fingerprint
            for point in points
        ):
            raise ValueError("frame score 中所有点击必须共享原帧尺寸和变换指纹")
        click_indices = tuple(
            sorted(evaluation.click_index for evaluation in self.result.clicks)
        )
        if click_indices != tuple(range(len(self.frame_clicks))):
            raise ValueError("canonical click evaluations 必须精确覆盖 frame_clicks")
        if not isinstance(self.unresolved_target_frame_indices, tuple):
            raise TypeError("unresolved_target_frame_indices 必须是 tuple")
        target_ids: list[str] = []
        for item in self.unresolved_target_frame_indices:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("unresolved target frame 记录必须是二元 tuple")
            target_id, frame_index = item
            if not isinstance(target_id, str) or not target_id:
                raise ValueError("unresolved target_id 必须是非空字符串")
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
            ):
                raise ValueError("unresolved frame_index 必须是非负整数")
            target_ids.append(target_id)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("unresolved target frame 记录不得重复 target_id")
        if tuple(target_ids) != tuple(sorted(target_ids)):
            raise ValueError("unresolved target frame 记录必须按 target_id 稳定排序")
        if any(
            target_id not in self.result.unresolved_target_ids
            for target_id in target_ids
        ):
            raise ValueError("frame 记录只能引用 canonical unresolved target")

    @property
    def clicks(self) -> tuple[ClickEvaluation, ...]:
        """返回 canonical 单击评分，兼容只读统计消费方。"""

        return self.result.clicks

    @property
    def resolved_targets(self) -> tuple[TargetResolution, ...]:
        """返回 canonical 已解析目标。"""

        return self.result.resolved_targets

    @property
    def unresolved_target_ids(self) -> tuple[str, ...]:
        """返回 canonical 未解析目标。"""

        return self.result.unresolved_target_ids

    @property
    def hit_count(self) -> int:
        """返回 canonical 命中数。"""

        return self.result.hit_count

    @property
    def miss_count(self) -> int:
        """返回 canonical miss 数。"""

        return self.result.miss_count

    @property
    def frequency_limited_count(self) -> int:
        """返回被频率限制的点击数。"""

        return self.result.frequency_limited_count


def _target_sort_key(target: TargetObject) -> tuple[float, int, str]:
    source_index = target.source_index
    return (
        target.start_ms,
        source_index if source_index is not None else 10**12,
        target.target_id,
    )


def _score_target(
    target: TargetObject,
    click: PredictedClick,
    *,
    circle_radius: float,
    spec: ScoreSpec,
) -> PointScore | SliderScore:
    if target.target_type == "circle":
        return score_point(
            (target.x or 0.0, target.y or 0.0),
            (click.x, click.y),
            circle_radius=circle_radius,
            reference_time_ms=target.start_ms,
            predicted_time_ms=click.time_ms,
            spec=spec,
        )
    predicted_path = click.path or ((click.x, click.y),)
    return score_slider(
        (target.x, target.y) if target.x is not None and target.y is not None else None,
        (click.x, click.y),
        target.path,
        predicted_path,
        circle_radius=circle_radius,
        reference_start_ms=target.start_ms,
        predicted_start_ms=click.time_ms,
        spec=spec,
    )


def _score_value(score: PointScore | SliderScore) -> float:
    return score.score.raw


def _spatial_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool:
    if isinstance(score, PointScore):
        return score.distance_ratio <= spec.spatial_pass_ratio
    return score.head.distance_ratio <= spec.spatial_pass_ratio and score.path.passed


def _temporal_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool:
    head = score if isinstance(score, PointScore) else score.head
    return head.time_error_ms <= spec.temporal_pass_end_ms


def _spatial_error(score: PointScore | SliderScore) -> float:
    return score.distance if isinstance(score, PointScore) else score.head.distance


def _temporal_error_ms(target: TargetObject, click: PredictedClick) -> float:
    return click.time_ms - target.start_ms


def _spatial_excess(score: PointScore | SliderScore, spec: ScoreSpec) -> float:
    head = score if isinstance(score, PointScore) else score.head
    ratio_excess = max(0.0, head.distance_ratio - spec.spatial_pass_ratio)
    denominator = spec.spatial_comfort_end_ratio - spec.spatial_pass_ratio
    return ratio_excess / denominator if denominator > 0 else ratio_excess


def _temporal_excess(score: PointScore | SliderScore, spec: ScoreSpec) -> float:
    head = score if isinstance(score, PointScore) else score.head
    time_excess = max(0.0, head.time_error_ms - spec.temporal_pass_end_ms)
    denominator = spec.temporal_comfort_end_ms - spec.temporal_pass_end_ms
    return time_excess / denominator if denominator > 0 else time_excess


def _error_attribution(
    target: TargetObject,
    click: PredictedClick,
    score: PointScore | SliderScore,
    *,
    spec: ScoreSpec,
) -> tuple[ErrorDomain, tuple[ErrorTag, ...], float, float]:
    spatial_passed = _spatial_passed(score, spec)
    temporal_passed = _temporal_passed(score, spec)
    spatial_error = _spatial_error(score)
    temporal_error = _temporal_error_ms(target, click)
    tags: list[ErrorTag] = []
    if not spatial_passed:
        if isinstance(score, SliderScore) and not score.path.passed:
            tags.append("slider_path_miss")
        if (
            isinstance(score, SliderScore)
            and score.head.distance_ratio > spec.spatial_pass_ratio
        ):
            tags.append("head_spatial_miss")
        if isinstance(score, PointScore):
            tags.append("spatial_miss")
    if not temporal_passed:
        tags.append("early_click" if temporal_error < 0 else "late_click")
    if spatial_passed and temporal_passed:
        return "none", (), spatial_error, temporal_error
    if spatial_passed:
        return "temporal", tuple(tags), spatial_error, temporal_error
    if temporal_passed:
        return "spatial", tuple(tags), spatial_error, temporal_error
    primary: ErrorDomain = (
        "temporal"
        if _temporal_excess(score, spec) >= _spatial_excess(score, spec)
        else "spatial"
    )
    return primary, tuple(tags), spatial_error, temporal_error


def _best_scored_target(
    targets: tuple[TargetObject, ...],
    click: PredictedClick,
    *,
    circle_radius: float,
    spec: ScoreSpec,
) -> tuple[TargetObject, PointScore | SliderScore] | None:
    scored = [
        (target, _score_target(target, click, circle_radius=circle_radius, spec=spec))
        for target in sorted(targets, key=_target_sort_key)
    ]
    if not scored:
        return None
    return max(scored, key=lambda item: (_score_value(item[1]), -item[0].start_ms))


def score_click_sequence(
    targets: tuple[TargetObject, ...],
    clicks: tuple[PredictedClick, ...],
    *,
    circle_radius: float,
    spec: SequenceScoreSpec = SequenceScoreSpec(),
) -> SequenceScore:
    """稳定排序点击，每个目标最多解析一次，并保留完整错误归因。"""

    if not isinstance(targets, tuple) or any(
        not isinstance(item, TargetObject) for item in targets
    ):
        raise TypeError("targets 必须是 TargetObject tuple")
    if not isinstance(clicks, tuple) or any(
        not isinstance(item, PredictedClick) for item in clicks
    ):
        raise TypeError("clicks 必须是 PredictedClick tuple")
    _finite("circle_radius", circle_radius)
    if circle_radius <= 0:
        raise ValueError("circle_radius 必须大于 0")
    if not isinstance(spec, SequenceScoreSpec):
        raise TypeError("spec 必须是 SequenceScoreSpec")
    active_targets = {
        target.target_id: target for target in sorted(targets, key=_target_sort_key)
    }
    if len(active_targets) != len(targets):
        raise ValueError("target_id 必须唯一")
    evaluations: list[ClickEvaluation] = []
    resolutions: list[TargetResolution] = []
    resolved_targets: dict[str, tuple[TargetObject, TargetResolution]] = {}
    last_accepted_click_ms: float | None = None
    ordered_clicks = sorted(
        enumerate(clicks), key=lambda item: (item[1].time_ms, item[0])
    )
    for click_index, click in ordered_clicks:
        if (
            last_accepted_click_ms is not None
            and click.time_ms - last_accepted_click_ms < spec.min_click_interval_ms
        ):
            evaluations.append(
                ClickEvaluation(
                    click_index,
                    click,
                    "frequency_limited",
                    primary_error="decision",
                    error_tags=("frequency_limited",),
                )
            )
            continue
        last_accepted_click_ms = click.time_ms
        passing: list[tuple[TargetObject, PointScore | SliderScore]] = []
        for target in sorted(active_targets.values(), key=_target_sort_key):
            score = _score_target(
                target, click, circle_radius=circle_radius, spec=spec.object_score_spec
            )
            if score.passed:
                passing.append((target, score))
        if not passing:
            duplicate = _best_scored_target(
                tuple(item[0] for item in resolved_targets.values()),
                click,
                circle_radius=circle_radius,
                spec=spec.object_score_spec,
            )
            if duplicate is not None and duplicate[1].passed:
                target, score = duplicate
                _, original = resolved_targets[target.target_id]
                tags: list[ErrorTag] = ["duplicate_after_hit"]
                if _score_value(score) > _score_value(original.score):
                    tags.append("better_score_after_resolution")
                evaluations.append(
                    ClickEvaluation(
                        click_index,
                        click,
                        "miss",
                        target.target_id,
                        target.source_index,
                        score,
                        "decision",
                        tuple(tags),
                        _spatial_error(score),
                        _temporal_error_ms(target, click),
                    )
                )
                continue
            best = _best_scored_target(
                tuple(active_targets.values()),
                click,
                circle_radius=circle_radius,
                spec=spec.object_score_spec,
            )
            if best is None:
                evaluations.append(
                    ClickEvaluation(
                        click_index,
                        click,
                        "miss",
                        primary_error="decision",
                        error_tags=("no_active_target",),
                    )
                )
                continue
            target, score = best
            primary_error, tags, spatial_error, temporal_error = _error_attribution(
                target, click, score, spec=spec.object_score_spec
            )
            evaluations.append(
                ClickEvaluation(
                    click_index,
                    click,
                    "miss",
                    target.target_id,
                    target.source_index,
                    score,
                    primary_error,
                    tags,
                    spatial_error,
                    temporal_error,
                )
            )
            continue
        target, score = passing[0]
        active_targets.pop(target.target_id)
        resolution = TargetResolution(
            target.target_id, target.source_index, click_index, click.time_ms, score
        )
        resolutions.append(resolution)
        resolved_targets[target.target_id] = (target, resolution)
        evaluations.append(
            ClickEvaluation(
                click_index,
                click,
                "hit",
                target.target_id,
                target.source_index,
                score,
                spatial_error=_spatial_error(score),
                temporal_error_ms=_temporal_error_ms(target, click),
            )
        )
    unresolved = tuple(
        target.target_id
        for target in sorted(active_targets.values(), key=_target_sort_key)
    )
    return SequenceScore(tuple(evaluations), tuple(resolutions), unresolved)


def score_frame_click_sequence(
    targets: tuple[TargetObject, ...],
    clicks: tuple[FramePredictedClick, ...],
    *,
    coordinate_transform: FrameCoordinateTransform,
    circle_radius: float,
    spec: SequenceScoreSpec = SequenceScoreSpec(),
) -> FrameSequenceScore:
    """先用共享指纹逆变换原帧点击，再委托唯一 canonical sequence scorer。"""

    if not isinstance(coordinate_transform, FrameCoordinateTransform):
        raise TypeError("coordinate_transform 必须是 FrameCoordinateTransform")
    if not isinstance(targets, tuple) or any(
        not isinstance(target, TargetObject) for target in targets
    ):
        raise TypeError("targets 必须是 TargetObject tuple")
    if not isinstance(clicks, tuple) or any(
        not isinstance(click, FramePredictedClick) for click in clicks
    ):
        raise TypeError("clicks 必须是 FramePredictedClick tuple")

    # 命中中心必须在 playfield 内；slider 的后续曲线控制点可以合法越界，
    # TargetObject 已保证它们是有限值，不能在这里裁剪或误判为坏样本。
    for target in targets:
        if target.x is not None and target.y is not None:
            OsuPoint(target.x, target.y)
        if target.target_type == "slider":
            OsuPoint(*target.path[0])

    canonical_clicks: list[PredictedClick] = []
    for click in clicks:
        position = coordinate_transform.prediction_to_canonical_scoring(click.position)
        canonical_path = tuple(
            coordinate_transform.prediction_to_canonical_scoring(point)
            for point in click.path
        )
        canonical_clicks.append(
            PredictedClick(
                time_ms=click.time_ms,
                x=position.x,
                y=position.y,
                path=tuple((point.x, point.y) for point in canonical_path),
            )
        )
    result = score_click_sequence(
        targets,
        tuple(canonical_clicks),
        circle_radius=circle_radius,
        spec=spec,
    )
    target_frame_indices = {
        target.target_id: target.frame_index
        for target in targets
        if target.frame_index is not None
    }
    return FrameSequenceScore(
        result=result,
        frame_clicks=clicks,
        source_frame_width=coordinate_transform.source_frame_width,
        source_frame_height=coordinate_transform.source_frame_height,
        transform_fingerprint=coordinate_transform.transform_fingerprint,
        unresolved_target_frame_indices=tuple(
            sorted(
                (target_id, target_frame_indices[target_id])
                for target_id in result.unresolved_target_ids
                if target_id in target_frame_indices
            )
        ),
    )


__all__ = (
    "ClickEvaluation",
    "ClickStatus",
    "ErrorDomain",
    "ErrorTag",
    "FramePredictedClick",
    "FrameSequenceScore",
    "PredictedClick",
    "SequenceScore",
    "SequenceScoreSpec",
    "TargetObject",
    "TargetResolution",
    "TargetType",
    "score_click_sequence",
    "score_frame_click_sequence",
)
