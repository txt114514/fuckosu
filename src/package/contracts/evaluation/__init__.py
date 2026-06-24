from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any

from package.contracts.base import ContractMixin
from package.contracts.geometry import Point2D


class PredictionAction(StrEnum):
    NO_OP = "no_op"
    PRESS = "press"
    HOLD = "hold"
    RELEASE = "release"


class ErrorDomain(StrEnum):
    NONE = "none"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    DECISION = "decision"


@dataclass(frozen=True)
class FrameRef(ContractMixin):
    sample_key: str
    frame_index: int
    timestamp_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.sample_key:
            raise ValueError("sample_key must not be empty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if self.timestamp_ms is not None and not isfinite(self.timestamp_ms):
            raise ValueError("timestamp_ms must be finite")


@dataclass(frozen=True)
class PredictionEvent(ContractMixin):
    action: PredictionAction
    point: Point2D | None = None
    time_ms: float | None = None
    candidate_id: int | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, PredictionAction):
            object.__setattr__(self, "action", PredictionAction(self.action))
        if self.point is not None and not isinstance(self.point, Point2D):
            object.__setattr__(self, "point", Point2D.from_mapping(self.point))
        if self.time_ms is not None and not isfinite(self.time_ms):
            raise ValueError("time_ms must be finite")
        if self.candidate_id is not None and self.candidate_id < 0:
            raise ValueError("candidate_id must be nonnegative")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in the 0..1 range")


@dataclass(frozen=True)
class ScoreSummary(ContractMixin):
    score_version: str
    quality_score: float
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.score_version:
            raise ValueError("score_version must not be empty")
        if not isfinite(self.quality_score) or not 0 <= self.quality_score <= 1:
            raise ValueError("quality_score must be in the 0..1 range")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be nonempty strings")
            if not isfinite(value):
                raise ValueError("metric values must be finite")


@dataclass(frozen=True)
class EvaluationOutcome(ContractMixin):
    frame: FrameRef
    passed: bool
    primary_error: ErrorDomain = ErrorDomain.NONE
    error_tags: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    prediction: PredictionEvent | None = None
    score: ScoreSummary | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame, FrameRef):
            object.__setattr__(self, "frame", FrameRef.from_mapping(self.frame))
        if not isinstance(self.primary_error, ErrorDomain):
            object.__setattr__(
                self,
                "primary_error",
                ErrorDomain(self.primary_error),
            )
        if self.prediction is not None and not isinstance(self.prediction, PredictionEvent):
            object.__setattr__(
                self,
                "prediction",
                PredictionEvent.from_mapping(self.prediction),
            )
        if self.score is not None and not isinstance(self.score, ScoreSummary):
            object.__setattr__(self, "score", ScoreSummary.from_mapping(self.score))
        object.__setattr__(self, "error_tags", tuple(self.error_tags))
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be nonempty strings")
            if not isfinite(value):
                raise ValueError("metric values must be finite")

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
    ) -> EvaluationOutcome:
        return cls(
            frame=data["frame"],
            passed=bool(data["passed"]),
            primary_error=data.get("primary_error", ErrorDomain.NONE),
            error_tags=tuple(data.get("error_tags", ())),
            metrics=dict(data.get("metrics", {})),
            prediction=data.get("prediction"),
            score=data.get("score"),
        )


__all__ = [
    "ErrorDomain",
    "EvaluationOutcome",
    "FrameRef",
    "PredictionAction",
    "PredictionEvent",
    "ScoreSummary",
]
