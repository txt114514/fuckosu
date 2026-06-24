from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from traning.lib.metrics import (
    PredictedClick,
    SequenceScore,
    SequenceScoreSpec,
    TargetObject,
    score_click_sequence,
)
from traning.state import TrialParameters


AGGREGATE_SCORE_VERSION = "point-slider-v2+click-sequence-v1+aggregate-v1"


@dataclass(frozen=True)
class TrialScoreSpec:
    miss_penalty_weight: float = 0.10
    frequency_penalty_weight: float = 0.08
    unresolved_penalty_weight: float = 0.35
    sample_pass_threshold: float = 0.75
    trial_pass_threshold: float = 0.80
    sequence_spec: SequenceScoreSpec = field(default_factory=SequenceScoreSpec)

    def __post_init__(self) -> None:
        values = (
            self.miss_penalty_weight,
            self.frequency_penalty_weight,
            self.unresolved_penalty_weight,
            self.sample_pass_threshold,
            self.trial_pass_threshold,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("trial score thresholds and weights must be finite")
        if self.sample_pass_threshold > 1 or self.trial_pass_threshold > 1:
            raise ValueError("pass thresholds must be in the 0..1 range")


@dataclass(frozen=True)
class SampleScoringInput:
    sample_key: str
    subproject: str
    targets: Sequence[TargetObject]
    predictions: Sequence[PredictedClick]
    circle_radius: float
    frame_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_key:
            raise ValueError("sample_key must not be empty")
        if not self.subproject:
            raise ValueError("subproject must not be empty")
        if not isfinite(self.circle_radius) or self.circle_radius <= 0:
            raise ValueError("circle_radius must be finite and positive")
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SampleScoreReport:
    sample_key: str
    subproject: str
    frame_index: int
    sequence: SequenceScore
    target_count: int
    click_count: int
    hit_count: int
    miss_count: int
    frequency_limited_count: int
    unresolved_count: int
    object_score: float
    quality_score: float
    passed: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_key": self.sample_key,
            "subproject": self.subproject,
            "frame_index": self.frame_index,
            "target_count": self.target_count,
            "click_count": self.click_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "frequency_limited_count": self.frequency_limited_count,
            "unresolved_count": self.unresolved_count,
            "object_score": self.object_score,
            "quality_score": self.quality_score,
            "passed": self.passed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrialScoreReport:
    trial_id: str
    score_version: str
    quality_score: float
    samples: tuple[SampleScoreReport, ...]
    pass_threshold: float
    parameters: TrialParameters = field(default_factory=TrialParameters)
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def target_count(self) -> int:
        return sum(sample.target_count for sample in self.samples)

    @property
    def hit_count(self) -> int:
        return sum(sample.hit_count for sample in self.samples)

    @property
    def miss_count(self) -> int:
        return sum(sample.miss_count for sample in self.samples)

    @property
    def unresolved_count(self) -> int:
        return sum(sample.unresolved_count for sample in self.samples)

    @property
    def frequency_limited_count(self) -> int:
        return sum(sample.frequency_limited_count for sample in self.samples)

    @property
    def passed(self) -> bool:
        return (
            all(sample.passed for sample in self.samples)
            and self.quality_score >= self.pass_threshold
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "score_version": self.score_version,
            "quality_score": self.quality_score,
            "target_count": self.target_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "unresolved_count": self.unresolved_count,
            "frequency_limited_count": self.frequency_limited_count,
            "passed": self.passed,
            "parameters": self.parameters.model_dump(mode="json"),
            "metrics": dict(self.metrics),
            "samples": [sample.as_dict() for sample in self.samples],
        }


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _safe_rate(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0


def _resolved_object_score(sequence: SequenceScore, target_count: int) -> float:
    if target_count == 0:
        return 1.0 if not sequence.clicks else 0.0
    score_sum = sum(
        resolution.score.score.normalized
        for resolution in sequence.resolved_targets
    )
    return _clamp01(score_sum / target_count)


def score_sample(
    sample: SampleScoringInput,
    *,
    spec: TrialScoreSpec = TrialScoreSpec(),
) -> SampleScoreReport:
    sequence = score_click_sequence(
        tuple(sample.targets),
        tuple(sample.predictions),
        circle_radius=sample.circle_radius,
        spec=spec.sequence_spec,
    )
    target_count = len(sample.targets)
    click_count = len(sample.predictions)
    miss_count = sequence.miss_count
    frequency_count = sequence.frequency_limited_count
    unresolved_count = len(sequence.unresolved_target_ids)
    object_score = _resolved_object_score(sequence, target_count)
    quality = _clamp01(
        object_score
        - spec.miss_penalty_weight * _safe_rate(miss_count, max(click_count, 1))
        - spec.frequency_penalty_weight
        * _safe_rate(frequency_count, max(click_count, 1))
        - spec.unresolved_penalty_weight
        * _safe_rate(unresolved_count, max(target_count, 1))
    )
    return SampleScoreReport(
        sample_key=sample.sample_key,
        subproject=sample.subproject,
        frame_index=sample.frame_index,
        sequence=sequence,
        target_count=target_count,
        click_count=click_count,
        hit_count=sequence.hit_count,
        miss_count=miss_count,
        frequency_limited_count=frequency_count,
        unresolved_count=unresolved_count,
        object_score=object_score,
        quality_score=quality,
        passed=quality >= spec.sample_pass_threshold and unresolved_count == 0,
        metadata=sample.metadata,
    )


def score_trial(
    trial_id: str,
    samples: Sequence[SampleScoringInput],
    *,
    parameters: TrialParameters | None = None,
    metrics: Mapping[str, float] | None = None,
    spec: TrialScoreSpec = TrialScoreSpec(),
) -> TrialScoreReport:
    if not trial_id:
        raise ValueError("trial_id must not be empty")
    reports = tuple(score_sample(sample, spec=spec) for sample in samples)
    if not reports:
        raise ValueError("trial scoring requires at least one sample")
    weighted_score_sum = sum(
        report.quality_score * max(report.target_count, 1)
        for report in reports
    )
    weight_sum = sum(max(report.target_count, 1) for report in reports)
    finite_metrics = {
        name: value
        for name, value in (metrics or {}).items()
        if isfinite(value)
    }
    return TrialScoreReport(
        trial_id=trial_id,
        score_version=AGGREGATE_SCORE_VERSION,
        quality_score=_clamp01(weighted_score_sum / weight_sum),
        samples=reports,
        pass_threshold=spec.trial_pass_threshold,
        parameters=parameters or TrialParameters(),
        metrics=finite_metrics,
    )


__all__ = [
    "AGGREGATE_SCORE_VERSION",
    "SampleScoreReport",
    "SampleScoringInput",
    "TrialScoreReport",
    "TrialScoreSpec",
    "score_sample",
    "score_trial",
]
