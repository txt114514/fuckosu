"""V2 唯一的点与 slider 连续空间/时间评分实现。"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import ceil, hypot, inf, isfinite, sqrt
from typing import TypeAlias


Point: TypeAlias = tuple[float, float]
PathPoints: TypeAlias = tuple[Point, ...]
SCORE_VERSION = "point-slider-v2"


def _require_number(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是数值")
    if not isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")
    if positive and value <= 0:
        raise ValueError(f"{name} 必须大于 0")


def _require_point(name: str, point: Point) -> None:
    if not isinstance(point, tuple) or len(point) != 2:
        raise TypeError(f"{name} 必须是二元坐标 tuple")
    for index, value in enumerate(point):
        _require_number(f"{name}[{index}]", value)


def _require_path(name: str, path: PathPoints, *, allow_empty: bool) -> None:
    if not isinstance(path, tuple):
        raise TypeError(f"{name} 必须是坐标 tuple")
    if not allow_empty and not path:
        raise ValueError(f"{name} 不得为空")
    for index, point in enumerate(path):
        _require_point(f"{name}[{index}]", point)


@dataclass(frozen=True, slots=True)
class ScoreSpec:
    """连续评分阈值；空间量以 circle radius 为单位，时间量为毫秒。"""

    spatial_bonus_max: float = 0.05
    spatial_bonus_clamp_ratio: float = 0.60
    spatial_pass_ratio: float = 1.00
    spatial_comfort_end_ratio: float = 1.50
    temporal_bonus_end_ms: float = 20.0
    temporal_full_end_ms: float = 50.0
    temporal_excellent_end_ms: float = 100.0
    temporal_pass_end_ms: float = 150.0
    temporal_comfort_end_ms: float = 200.0
    temporal_excellent_score: float = 0.80
    temporal_pass_score: float = 0.50
    comfort_score_max: float = 0.05
    slider_path_pass_ratio: float = 1.50
    slider_path_sample_step_ratio: float = 0.25

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_number(field.name, getattr(self, field.name))
        nonnegative = (
            self.spatial_bonus_max,
            self.spatial_bonus_clamp_ratio,
            self.temporal_bonus_end_ms,
            self.temporal_excellent_score,
            self.temporal_pass_score,
            self.comfort_score_max,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("评分 bonus、阈值与系数不得为负数")
        ordered_spatial = (
            self.spatial_bonus_clamp_ratio,
            self.spatial_pass_ratio,
            self.spatial_comfort_end_ratio,
        )
        ordered_temporal = (
            self.temporal_bonus_end_ms,
            self.temporal_full_end_ms,
            self.temporal_excellent_end_ms,
            self.temporal_pass_end_ms,
            self.temporal_comfort_end_ms,
        )
        if any(
            left >= right for left, right in zip(ordered_spatial, ordered_spatial[1:])
        ):
            raise ValueError("空间阈值必须严格递增")
        if any(
            left >= right for left, right in zip(ordered_temporal, ordered_temporal[1:])
        ):
            raise ValueError("时间阈值必须严格递增")
        if self.slider_path_pass_ratio <= 0 or self.slider_path_sample_step_ratio <= 0:
            raise ValueError("slider 路径阈值和采样步长必须大于 0")

    @property
    def maximum_coefficient(self) -> float:
        """返回包含空间 bonus 后单个系数的理论上限。"""

        return 1.0 + self.spatial_bonus_max

    @property
    def maximum_raw_score(self) -> float:
        """返回空间、时间及交互项组合后的理论原始分上限。"""

        maximum = self.maximum_coefficient
        return maximum + maximum + maximum * maximum


@dataclass(frozen=True, slots=True)
class CombinedScore:
    """组合后的空间、时间、原始和归一化分数。"""

    spatial: float
    temporal: float
    raw: float
    normalized: float


@dataclass(frozen=True, slots=True)
class PointScore:
    """单点连续评分。"""

    distance: float
    distance_ratio: float
    time_error_ms: float
    score: CombinedScore
    passed: bool


@dataclass(frozen=True, slots=True)
class PathScore:
    """slider 路径的双向走廊评分。"""

    dilation_radius: float
    reference_coverage: float
    prediction_precision: float
    reference_max_distance_ratio: float
    prediction_max_distance_ratio: float
    coefficient: float
    passed: bool


@dataclass(frozen=True, slots=True)
class SliderScore:
    """slider head 与 path 的联合评分。"""

    head: PointScore
    path: PathScore
    score: CombinedScore
    passed: bool


def _require_spec(spec: ScoreSpec) -> None:
    if not isinstance(spec, ScoreSpec):
        raise TypeError("spec 必须是 ScoreSpec")


def _interpolate(
    value: float, start: float, end: float, start_score: float, end_score: float
) -> float:
    progress = (value - start) / (end - start)
    return start_score + progress * (end_score - start_score)


def spatial_coefficient(
    distance_ratio: float, *, spec: ScoreSpec = ScoreSpec()
) -> float:
    """把非负距离半径比映射为连续空间系数。"""

    _require_spec(spec)
    _require_number("distance_ratio", distance_ratio)
    if distance_ratio < 0:
        raise ValueError("distance_ratio 不得为负数")
    if distance_ratio <= spec.spatial_pass_ratio:
        clamped = max(distance_ratio, spec.spatial_bonus_clamp_ratio)
        bonus_progress = (spec.spatial_pass_ratio - clamped) / (
            spec.spatial_pass_ratio - spec.spatial_bonus_clamp_ratio
        )
        return 1.0 + spec.spatial_bonus_max * sqrt(bonus_progress)
    if distance_ratio < spec.spatial_comfort_end_ratio:
        comfort_progress = (spec.spatial_comfort_end_ratio - distance_ratio) / (
            spec.spatial_comfort_end_ratio - spec.spatial_pass_ratio
        )
        return spec.comfort_score_max * comfort_progress**2
    return 0.0


def temporal_coefficient(
    time_error_ms: float, *, spec: ScoreSpec = ScoreSpec()
) -> float:
    """按绝对时间误差分段插值为连续时间系数。"""

    _require_spec(spec)
    _require_number("time_error_ms", time_error_ms)
    error = abs(time_error_ms)
    maximum = spec.maximum_coefficient
    if error <= spec.temporal_bonus_end_ms:
        return maximum
    if error <= spec.temporal_full_end_ms:
        return _interpolate(
            error, spec.temporal_bonus_end_ms, spec.temporal_full_end_ms, maximum, 1.0
        )
    if error <= spec.temporal_excellent_end_ms:
        return _interpolate(
            error,
            spec.temporal_full_end_ms,
            spec.temporal_excellent_end_ms,
            1.0,
            spec.temporal_excellent_score,
        )
    if error <= spec.temporal_pass_end_ms:
        return _interpolate(
            error,
            spec.temporal_excellent_end_ms,
            spec.temporal_pass_end_ms,
            spec.temporal_excellent_score,
            spec.temporal_pass_score,
        )
    if error < spec.temporal_comfort_end_ms:
        comfort_progress = (spec.temporal_comfort_end_ms - error) / (
            spec.temporal_comfort_end_ms - spec.temporal_pass_end_ms
        )
        return spec.comfort_score_max * comfort_progress**2
    return 0.0


def combine_coefficients(
    spatial: float, temporal: float, *, spec: ScoreSpec = ScoreSpec()
) -> CombinedScore:
    """组合空间与时间系数，并按理论最大值归一化。"""

    _require_spec(spec)
    _require_number("spatial", spatial)
    _require_number("temporal", temporal)
    if spatial < 0 or temporal < 0:
        raise ValueError("评分系数不得为负数")
    raw = spatial + temporal + spatial * temporal
    return CombinedScore(spatial, temporal, raw, raw / spec.maximum_raw_score)


def score_point(
    reference_xy: Point,
    predicted_xy: Point,
    *,
    circle_radius: float,
    reference_time_ms: float,
    predicted_time_ms: float,
    spec: ScoreSpec = ScoreSpec(),
) -> PointScore:
    """在同一坐标空间内评分点位置和毫秒级打击时间。"""

    _require_spec(spec)
    _require_point("reference_xy", reference_xy)
    _require_point("predicted_xy", predicted_xy)
    _require_number("circle_radius", circle_radius, positive=True)
    _require_number("reference_time_ms", reference_time_ms)
    _require_number("predicted_time_ms", predicted_time_ms)
    distance = hypot(
        predicted_xy[0] - reference_xy[0], predicted_xy[1] - reference_xy[1]
    )
    distance_ratio = distance / circle_radius
    time_error_ms = abs(predicted_time_ms - reference_time_ms)
    score = combine_coefficients(
        spatial_coefficient(distance_ratio, spec=spec),
        temporal_coefficient(time_error_ms, spec=spec),
        spec=spec,
    )
    return PointScore(
        distance,
        distance_ratio,
        time_error_ms,
        score,
        distance_ratio <= spec.spatial_pass_ratio
        and time_error_ms <= spec.temporal_pass_end_ms,
    )


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_squared = segment_x**2 + segment_y**2
    if length_squared == 0:
        return hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * segment_x + (point[1] - start[1]) * segment_y
    ) / length_squared
    clamped = min(1.0, max(0.0, projection))
    nearest = (start[0] + clamped * segment_x, start[1] + clamped * segment_y)
    return hypot(point[0] - nearest[0], point[1] - nearest[1])


def _minimum_distance(point: Point, path: PathPoints) -> float:
    if len(path) == 1:
        return hypot(point[0] - path[0][0], point[1] - path[0][1])
    return min(
        _point_to_segment_distance(point, start, end)
        for start, end in zip(path, path[1:])
    )


def _densify_path(path: PathPoints, *, maximum_step: float) -> PathPoints:
    if len(path) <= 1:
        return path
    dense: list[Point] = [path[0]]
    for start, end in zip(path, path[1:]):
        length = hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, ceil(length / maximum_step))
        dense.extend(
            (
                start[0] + (end[0] - start[0]) * index / steps,
                start[1] + (end[1] - start[1]) * index / steps,
            )
            for index in range(1, steps + 1)
        )
    return tuple(dense)


def _directed_path_statistics(
    source: PathPoints, target: PathPoints, *, distance_limit: float
) -> tuple[float, float]:
    if not source:
        return 0.0, inf
    distances = tuple(_minimum_distance(point, target) for point in source)
    return sum(distance <= distance_limit for distance in distances) / len(
        distances
    ), max(distances)


def score_slider_path(
    reference_path: PathPoints,
    predicted_path: PathPoints,
    *,
    circle_radius: float,
    spec: ScoreSpec = ScoreSpec(),
) -> PathScore:
    """用双向稠密采样评估 slider 中心线膨胀走廊。"""

    _require_spec(spec)
    _require_path("reference_path", reference_path, allow_empty=False)
    _require_path("predicted_path", predicted_path, allow_empty=True)
    _require_number("circle_radius", circle_radius, positive=True)
    if not predicted_path:
        return PathScore(
            circle_radius * spec.slider_path_pass_ratio, 0.0, 0.0, inf, inf, 0.0, False
        )
    dilation_radius = circle_radius * spec.slider_path_pass_ratio
    sample_step = circle_radius * spec.slider_path_sample_step_ratio
    sampled_reference = _densify_path(reference_path, maximum_step=sample_step)
    sampled_prediction = _densify_path(predicted_path, maximum_step=sample_step)
    coverage, reference_max_distance = _directed_path_statistics(
        sampled_reference, sampled_prediction, distance_limit=dilation_radius
    )
    precision, prediction_max_distance = _directed_path_statistics(
        sampled_prediction, sampled_reference, distance_limit=dilation_radius
    )
    reference_ratio = reference_max_distance / circle_radius
    prediction_ratio = prediction_max_distance / circle_radius
    worst_ratio = max(reference_ratio, prediction_ratio)
    passed = worst_ratio <= spec.slider_path_pass_ratio
    if passed:
        clamped = max(worst_ratio, spec.spatial_bonus_clamp_ratio)
        bonus_progress = (spec.slider_path_pass_ratio - clamped) / (
            spec.slider_path_pass_ratio - spec.spatial_bonus_clamp_ratio
        )
        coefficient = 1.0 + spec.spatial_bonus_max * sqrt(bonus_progress)
    else:
        coefficient = spec.comfort_score_max * coverage * precision
    return PathScore(
        dilation_radius,
        coverage,
        precision,
        reference_ratio,
        prediction_ratio,
        coefficient,
        passed,
    )


def score_slider(
    reference_head_xy: Point | None,
    predicted_head_xy: Point | None,
    reference_path: PathPoints,
    predicted_path: PathPoints,
    *,
    circle_radius: float,
    reference_start_ms: float,
    predicted_start_ms: float,
    spec: ScoreSpec = ScoreSpec(),
) -> SliderScore:
    """联合评分 slider head、路径与开始时间。"""

    _require_spec(spec)
    _require_path("reference_path", reference_path, allow_empty=True)
    _require_path("predicted_path", predicted_path, allow_empty=True)
    if reference_head_xy is None:
        if not reference_path:
            raise ValueError("reference head 需要 reference_head_xy 或 reference_path")
        reference_head_xy = reference_path[0]
    else:
        _require_point("reference_head_xy", reference_head_xy)
    if predicted_head_xy is None:
        if not predicted_path:
            raise ValueError("predicted head 需要 predicted_head_xy 或 predicted_path")
        predicted_head_xy = predicted_path[0]
    else:
        _require_point("predicted_head_xy", predicted_head_xy)
    head = score_point(
        reference_head_xy,
        predicted_head_xy,
        circle_radius=circle_radius,
        reference_time_ms=reference_start_ms,
        predicted_time_ms=predicted_start_ms,
        spec=spec,
    )
    path = score_slider_path(
        reference_path, predicted_path, circle_radius=circle_radius, spec=spec
    )
    score = combine_coefficients(
        min(head.score.spatial, path.coefficient), head.score.temporal, spec=spec
    )
    return SliderScore(head, path, score, head.passed and path.passed)


__all__ = (
    "CombinedScore",
    "PathPoints",
    "PathScore",
    "Point",
    "PointScore",
    "SCORE_VERSION",
    "ScoreSpec",
    "SliderScore",
    "combine_coefficients",
    "score_point",
    "score_slider",
    "score_slider_path",
    "spatial_coefficient",
    "temporal_coefficient",
)
