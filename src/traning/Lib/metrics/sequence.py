from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

from traning.Lib.metrics.scoring import (
    PathPoints,
    Point,
    PointScore,
    ScoreSpec,
    SliderScore,
    score_point,
    score_slider,
)


TargetType = Literal["circle", "slider"]
ClickStatus = Literal["hit", "miss", "frequency_limited"]
ErrorDomain = Literal["none", "spatial", "temporal", "decision"]
ErrorTag = Literal[
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


@dataclass(frozen=True)
class SequenceScoreSpec:
    min_click_interval_ms: float = 50.0
    object_score_spec: ScoreSpec = field(default_factory=ScoreSpec)

    def __post_init__(self) -> None:
        if (
            not isfinite(self.min_click_interval_ms)
            or self.min_click_interval_ms < 0
        ):
            raise ValueError("min_click_interval_ms must be finite and nonnegative")


@dataclass(frozen=True)
class TargetObject:
    target_id: str
    target_type: TargetType
    start_ms: float
    end_ms: float
    x: float | None = None
    y: float | None = None
    path: PathPoints = ()
    source_index: int | None = None

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must not be empty")
        if not isfinite(self.start_ms) or not isfinite(self.end_ms):
            raise ValueError("target times must be finite")
        if self.end_ms < self.start_ms:
            raise ValueError("target end_ms must not be earlier than start_ms")
        if self.target_type == "circle" and (self.x is None or self.y is None):
            raise ValueError("circle targets require x and y")
        if self.target_type == "slider" and not self.path:
            raise ValueError("slider targets require a path")


@dataclass(frozen=True)
class PredictedClick:
    time_ms: float
    x: float
    y: float
    path: PathPoints = ()

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.time_ms, self.x, self.y)):
            raise ValueError("click time and coordinates must be finite")


@dataclass(frozen=True)
class TargetResolution:
    target_id: str
    source_index: int | None
    click_index: int
    click_time_ms: float
    score: PointScore | SliderScore


@dataclass(frozen=True)
class ClickEvaluation:
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
        return self.status == "frequency_limited"


@dataclass(frozen=True)
class SequenceScore:
    clicks: tuple[ClickEvaluation, ...]
    resolved_targets: tuple[TargetResolution, ...]
    unresolved_target_ids: tuple[str, ...]

    @property
    def hit_count(self) -> int:
        return len(self.resolved_targets)

    @property
    def miss_count(self) -> int:
        return sum(item.status == "miss" for item in self.clicks)

    @property
    def frequency_limited_count(self) -> int:
        return sum(item.status == "frequency_limited" for item in self.clicks)


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
    return (
        score.head.distance_ratio <= spec.spatial_pass_ratio
        and score.path.passed
    )


def _temporal_passed(score: PointScore | SliderScore, spec: ScoreSpec) -> bool:
    head = score if isinstance(score, PointScore) else score.head
    return head.time_error_ms <= spec.temporal_pass_end_ms


def _spatial_error(score: PointScore | SliderScore) -> float:
    if isinstance(score, PointScore):
        return score.distance
    return score.head.distance


def _temporal_error_ms(
    target: TargetObject,
    click: PredictedClick,
) -> float:
    return click.time_ms - target.start_ms


def _spatial_excess(
    score: PointScore | SliderScore,
    spec: ScoreSpec,
) -> float:
    head = score if isinstance(score, PointScore) else score.head
    ratio_excess = max(0.0, head.distance_ratio - spec.spatial_pass_ratio)
    denominator = spec.spatial_comfort_end_ratio - spec.spatial_pass_ratio
    return ratio_excess / denominator if denominator > 0 else ratio_excess


def _temporal_excess(
    score: PointScore | SliderScore,
    spec: ScoreSpec,
) -> float:
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
        if isinstance(score, SliderScore) and (
            score.head.distance_ratio > spec.spatial_pass_ratio
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
        (
            target,
            _score_target(
                target,
                click,
                circle_radius=circle_radius,
                spec=spec,
            ),
        )
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
    if not isfinite(circle_radius) or circle_radius <= 0:
        raise ValueError("circle_radius must be finite and positive")

    active_targets = {
        target.target_id: target
        for target in sorted(targets, key=_target_sort_key)
    }
    if len(active_targets) != len(targets):
        raise ValueError("target_id values must be unique")

    evaluations: list[ClickEvaluation] = []
    resolutions: list[TargetResolution] = []
    resolved_targets: dict[str, tuple[TargetObject, TargetResolution]] = {}
    last_accepted_click_ms: float | None = None
    ordered_clicks = sorted(
        enumerate(clicks),
        key=lambda item: (item[1].time_ms, item[0]),
    )

    for click_index, click in ordered_clicks:
        if (
            last_accepted_click_ms is not None
            and click.time_ms - last_accepted_click_ms
            < spec.min_click_interval_ms
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
                target,
                click,
                circle_radius=circle_radius,
                spec=spec.object_score_spec,
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
                        target_id=target.target_id,
                        source_index=target.source_index,
                        score=score,
                        primary_error="decision",
                        error_tags=tuple(tags),
                        spatial_error=_spatial_error(score),
                        temporal_error_ms=_temporal_error_ms(target, click),
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
            primary_error, tags, spatial_error, temporal_error = (
                _error_attribution(
                    target,
                    click,
                    score,
                    spec=spec.object_score_spec,
                )
            )
            evaluations.append(
                ClickEvaluation(
                    click_index,
                    click,
                    "miss",
                    target_id=target.target_id,
                    source_index=target.source_index,
                    score=score,
                    primary_error=primary_error,
                    error_tags=tags,
                    spatial_error=spatial_error,
                    temporal_error_ms=temporal_error,
                )
            )
            continue

        target, score = passing[0]
        active_targets.pop(target.target_id)
        resolution = TargetResolution(
            target_id=target.target_id,
            source_index=target.source_index,
            click_index=click_index,
            click_time_ms=click.time_ms,
            score=score,
        )
        resolutions.append(resolution)
        resolved_targets[target.target_id] = (target, resolution)
        evaluations.append(
            ClickEvaluation(
                click_index,
                click,
                "hit",
                target_id=target.target_id,
                source_index=target.source_index,
                score=score,
                spatial_error=_spatial_error(score),
                temporal_error_ms=_temporal_error_ms(target, click),
            )
        )

    unresolved = tuple(
        target.target_id
        for target in sorted(active_targets.values(), key=_target_sort_key)
    )
    return SequenceScore(
        clicks=tuple(evaluations),
        resolved_targets=tuple(resolutions),
        unresolved_target_ids=unresolved,
    )


__all__ = [
    "ClickEvaluation",
    "ClickStatus",
    "ErrorDomain",
    "ErrorTag",
    "PredictedClick",
    "SequenceScore",
    "SequenceScoreSpec",
    "TargetObject",
    "TargetResolution",
    "TargetType",
    "score_click_sequence",
]
