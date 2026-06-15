from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot, inf, isfinite, sqrt


Point = tuple[float, float]
PathPoints = tuple[Point, ...]
SCORE_VERSION = "point-slider-v2"


@dataclass(frozen=True)
class ScoreSpec:
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
        if any(not isfinite(value) for value in (*ordered_spatial, *ordered_temporal)):
            raise ValueError("score thresholds must be finite")
        if not ordered_spatial[0] < ordered_spatial[1] < ordered_spatial[2]:
            raise ValueError("spatial thresholds must be strictly increasing")
        if any(
            left >= right
            for left, right in zip(ordered_temporal, ordered_temporal[1:])
        ):
            raise ValueError("temporal thresholds must be strictly increasing")
        if self.slider_path_pass_ratio <= 0:
            raise ValueError("slider_path_pass_ratio must be positive")
        if self.slider_path_sample_step_ratio <= 0:
            raise ValueError(
                "slider_path_sample_step_ratio must be positive"
            )

    @property
    def maximum_coefficient(self) -> float:
        return 1.0 + self.spatial_bonus_max

    @property
    def maximum_raw_score(self) -> float:
        maximum = self.maximum_coefficient
        return maximum + maximum + maximum * maximum


@dataclass(frozen=True)
class CombinedScore:
    spatial: float
    temporal: float
    raw: float
    normalized: float


@dataclass(frozen=True)
class PointScore:
    distance: float
    distance_ratio: float
    time_error_ms: float
    score: CombinedScore
    passed: bool


@dataclass(frozen=True)
class PathScore:
    dilation_radius: float
    reference_coverage: float
    prediction_precision: float
    reference_max_distance_ratio: float
    prediction_max_distance_ratio: float
    coefficient: float
    passed: bool


@dataclass(frozen=True)
class SliderScore:
    head: PointScore
    path: PathScore
    score: CombinedScore
    passed: bool


def _interpolate(
    value: float,
    start: float,
    end: float,
    start_score: float,
    end_score: float,
) -> float:
    progress = (value - start) / (end - start)
    return start_score + progress * (end_score - start_score)


def spatial_coefficient(
    distance_ratio: float,
    *,
    spec: ScoreSpec = ScoreSpec(),
) -> float:
    if not isfinite(distance_ratio) or distance_ratio < 0:
        raise ValueError("distance_ratio must be finite and nonnegative")
    if distance_ratio <= spec.spatial_pass_ratio:
        clamped = max(distance_ratio, spec.spatial_bonus_clamp_ratio)
        bonus_progress = (
            spec.spatial_pass_ratio - clamped
        ) / (
            spec.spatial_pass_ratio - spec.spatial_bonus_clamp_ratio
        )
        return 1.0 + spec.spatial_bonus_max * sqrt(bonus_progress)
    if distance_ratio < spec.spatial_comfort_end_ratio:
        comfort_progress = (
            spec.spatial_comfort_end_ratio - distance_ratio
        ) / (
            spec.spatial_comfort_end_ratio - spec.spatial_pass_ratio
        )
        return spec.comfort_score_max * comfort_progress**2
    return 0.0


def temporal_coefficient(
    time_error_ms: float,
    *,
    spec: ScoreSpec = ScoreSpec(),
) -> float:
    error = abs(time_error_ms)
    if not isfinite(error):
        raise ValueError("time_error_ms must be finite")
    maximum = spec.maximum_coefficient
    if error <= spec.temporal_bonus_end_ms:
        return maximum
    if error <= spec.temporal_full_end_ms:
        return _interpolate(
            error,
            spec.temporal_bonus_end_ms,
            spec.temporal_full_end_ms,
            maximum,
            1.0,
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
        comfort_progress = (
            spec.temporal_comfort_end_ms - error
        ) / (
            spec.temporal_comfort_end_ms - spec.temporal_pass_end_ms
        )
        return spec.comfort_score_max * comfort_progress**2
    return 0.0


def combine_coefficients(
    spatial: float,
    temporal: float,
    *,
    spec: ScoreSpec = ScoreSpec(),
) -> CombinedScore:
    if not isfinite(spatial) or not isfinite(temporal):
        raise ValueError("score coefficients must be finite")
    if spatial < 0 or temporal < 0:
        raise ValueError("score coefficients must be nonnegative")
    raw = spatial + temporal + spatial * temporal
    return CombinedScore(
        spatial=spatial,
        temporal=temporal,
        raw=raw,
        normalized=raw / spec.maximum_raw_score,
    )


def score_point(
    reference_xy: Point,
    predicted_xy: Point,
    *,
    circle_radius: float,
    reference_time_ms: float,
    predicted_time_ms: float,
    spec: ScoreSpec = ScoreSpec(),
) -> PointScore:
    if not isfinite(circle_radius) or circle_radius <= 0:
        raise ValueError("circle_radius must be finite and positive")
    distance = hypot(
        predicted_xy[0] - reference_xy[0],
        predicted_xy[1] - reference_xy[1],
    )
    distance_ratio = distance / circle_radius
    time_error_ms = abs(predicted_time_ms - reference_time_ms)
    score = combine_coefficients(
        spatial_coefficient(distance_ratio, spec=spec),
        temporal_coefficient(time_error_ms, spec=spec),
        spec=spec,
    )
    return PointScore(
        distance=distance,
        distance_ratio=distance_ratio,
        time_error_ms=time_error_ms,
        score=score,
        passed=(
            distance_ratio <= spec.spatial_pass_ratio
            and time_error_ms <= spec.temporal_pass_end_ms
        ),
    )


def _point_to_segment_distance(
    point: Point,
    start: Point,
    end: Point,
) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_squared = segment_x**2 + segment_y**2
    if length_squared == 0:
        return hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * segment_x
        + (point[1] - start[1]) * segment_y
    ) / length_squared
    clamped = min(1.0, max(0.0, projection))
    nearest = (
        start[0] + clamped * segment_x,
        start[1] + clamped * segment_y,
    )
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
    source: PathPoints,
    target: PathPoints,
    *,
    distance_limit: float,
) -> tuple[float, float]:
    """Measure source centerline samples inside the dilated target corridor."""
    if not source:
        return 0.0, inf
    distances = tuple(_minimum_distance(point, target) for point in source)
    within_limit = sum(distance <= distance_limit for distance in distances)
    return within_limit / len(distances), max(distances)


def score_slider_path(
    reference_path: PathPoints,
    predicted_path: PathPoints,
    *,
    circle_radius: float,
    spec: ScoreSpec = ScoreSpec(),
) -> PathScore:
    if not reference_path:
        raise ValueError("reference_path must not be empty")
    if not isfinite(circle_radius) or circle_radius <= 0:
        raise ValueError("circle_radius must be finite and positive")
    if not predicted_path:
        return PathScore(
            dilation_radius=(
                circle_radius * spec.slider_path_pass_ratio
            ),
            reference_coverage=0.0,
            prediction_precision=0.0,
            reference_max_distance_ratio=inf,
            prediction_max_distance_ratio=inf,
            coefficient=0.0,
            passed=False,
        )

    dilation_radius = circle_radius * spec.slider_path_pass_ratio
    sample_step = circle_radius * spec.slider_path_sample_step_ratio
    sampled_reference = _densify_path(
        reference_path,
        maximum_step=sample_step,
    )
    sampled_prediction = _densify_path(
        predicted_path,
        maximum_step=sample_step,
    )
    # A point is inside a polyline dilated by radius R exactly when its
    # shortest distance to that polyline is at most R. Dense path sampling
    # therefore implements the requested 1.5x corridor without rasterizing.
    coverage, reference_max_distance = _directed_path_statistics(
        sampled_reference,
        sampled_prediction,
        distance_limit=dilation_radius,
    )
    precision, prediction_max_distance = _directed_path_statistics(
        sampled_prediction,
        sampled_reference,
        distance_limit=dilation_radius,
    )
    reference_ratio = reference_max_distance / circle_radius
    prediction_ratio = prediction_max_distance / circle_radius
    worst_ratio = max(reference_ratio, prediction_ratio)
    passed = worst_ratio <= spec.slider_path_pass_ratio
    if passed:
        clamped = max(worst_ratio, spec.spatial_bonus_clamp_ratio)
        bonus_progress = (
            spec.slider_path_pass_ratio - clamped
        ) / (
            spec.slider_path_pass_ratio - spec.spatial_bonus_clamp_ratio
        )
        coefficient = 1.0 + spec.spatial_bonus_max * sqrt(bonus_progress)
    else:
        coefficient = spec.comfort_score_max * coverage * precision
    return PathScore(
        dilation_radius=dilation_radius,
        reference_coverage=coverage,
        prediction_precision=precision,
        reference_max_distance_ratio=reference_ratio,
        prediction_max_distance_ratio=prediction_ratio,
        coefficient=coefficient,
        passed=passed,
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
    if reference_head_xy is None:
        if not reference_path:
            raise ValueError(
                "reference head requires reference_head_xy or reference_path"
            )
        reference_head_xy = reference_path[0]
    if predicted_head_xy is None:
        if not predicted_path:
            raise ValueError(
                "predicted head requires predicted_head_xy or predicted_path"
            )
        predicted_head_xy = predicted_path[0]
    head = score_point(
        reference_head_xy,
        predicted_head_xy,
        circle_radius=circle_radius,
        reference_time_ms=reference_start_ms,
        predicted_time_ms=predicted_start_ms,
        spec=spec,
    )
    path = score_slider_path(
        reference_path,
        predicted_path,
        circle_radius=circle_radius,
        spec=spec,
    )
    spatial = min(head.score.spatial, path.coefficient)
    score = combine_coefficients(
        spatial,
        head.score.temporal,
        spec=spec,
    )
    return SliderScore(
        head=head,
        path=path,
        score=score,
        passed=head.passed and path.passed,
    )


__all__ = [
    "CombinedScore",
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
]
