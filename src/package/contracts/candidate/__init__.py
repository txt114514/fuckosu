from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from package.contracts.base import ContractMixin
from package.contracts.evaluation import PredictionAction
from package.contracts.geometry import CoordinateSpace, Point2D, Rect2D


@dataclass(frozen=True)
class SpatialCandidateRef(ContractMixin):
    candidate_id: int
    point: Point2D
    score: float
    object_type: str
    embedding: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.candidate_id < 0:
            raise ValueError("candidate_id must be nonnegative")
        if not isinstance(self.point, Point2D):
            object.__setattr__(self, "point", Point2D.from_mapping(self.point))
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("candidate score must be in the 0..1 range")
        if not self.object_type:
            raise ValueError("object_type must not be empty")
        if any(not isfinite(value) for value in self.embedding):
            raise ValueError("embedding values must be finite")
        object.__setattr__(self, "embedding", tuple(float(value) for value in self.embedding))


@dataclass(frozen=True)
class SliderPathCandidateRef(ContractMixin):
    path_id: int
    polyline: tuple[Point2D, ...]
    score: float
    bbox: Rect2D | None = None
    ambiguous: bool = False
    ambiguity_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.path_id < 0:
            raise ValueError("path_id must be nonnegative")
        points = tuple(
            point if isinstance(point, Point2D) else Point2D.from_mapping(point)
            for point in self.polyline
        )
        if len(points) < 2:
            raise ValueError("slider path candidates require at least two points")
        object.__setattr__(self, "polyline", points)
        if self.bbox is not None and not isinstance(self.bbox, Rect2D):
            object.__setattr__(self, "bbox", Rect2D.from_mapping(self.bbox))
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("path score must be in the 0..1 range")
        object.__setattr__(self, "ambiguity_reasons", tuple(self.ambiguity_reasons))


@dataclass(frozen=True)
class TemporalTargetRef(ContractMixin):
    action: PredictionAction
    target_time_ms: float | None = None
    candidate_id: int | None = None
    point: Point2D | None = None
    strategy: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.action, PredictionAction):
            object.__setattr__(self, "action", PredictionAction(self.action))
        if self.target_time_ms is not None and not isfinite(self.target_time_ms):
            raise ValueError("target_time_ms must be finite")
        if self.candidate_id is not None and self.candidate_id < 0:
            raise ValueError("candidate_id must be nonnegative")
        if self.point is not None and not isinstance(self.point, Point2D):
            object.__setattr__(self, "point", Point2D.from_mapping(self.point))
        if not self.strategy:
            raise ValueError("strategy must not be empty")


@dataclass(frozen=True)
class CandidateCacheFrameRef(ContractMixin):
    version: str
    sample_key: str
    frame_index: int
    timestamp_ms: float
    candidates: tuple[SpatialCandidateRef, ...] = ()
    slider_paths: tuple[SliderPathCandidateRef, ...] = ()
    temporal_target: TemporalTargetRef | None = None

    def __post_init__(self) -> None:
        if not self.version or not self.sample_key:
            raise ValueError("cache frame version and sample_key must not be empty")
        if self.frame_index < 0 or self.timestamp_ms < 0:
            raise ValueError("frame index and timestamp must be nonnegative")
        object.__setattr__(
            self,
            "candidates",
            tuple(
                item
                if isinstance(item, SpatialCandidateRef)
                else SpatialCandidateRef.from_mapping(item)
                for item in self.candidates
            ),
        )
        object.__setattr__(
            self,
            "slider_paths",
            tuple(
                item
                if isinstance(item, SliderPathCandidateRef)
                else SliderPathCandidateRef.from_mapping(item)
                for item in self.slider_paths
            ),
        )
        if self.temporal_target is not None and not isinstance(self.temporal_target, TemporalTargetRef):
            object.__setattr__(
                self,
                "temporal_target",
                TemporalTargetRef.from_mapping(self.temporal_target),
            )


@dataclass(frozen=True)
class DecisionFrameRecord(ContractMixin):
    version: str
    sample_key: str
    frame_index: int
    action: PredictionAction
    action_probability: float
    point: Point2D | None = None
    candidate_id: int | None = None

    def __post_init__(self) -> None:
        if not self.version or not self.sample_key:
            raise ValueError("decision version and sample_key must not be empty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if not isinstance(self.action, PredictionAction):
            object.__setattr__(self, "action", PredictionAction(self.action))
        if not isfinite(self.action_probability) or not 0 <= self.action_probability <= 1:
            raise ValueError("action_probability must be in the 0..1 range")
        if self.point is not None and not isinstance(self.point, Point2D):
            object.__setattr__(self, "point", Point2D.from_mapping(self.point))
        if self.candidate_id is not None and self.candidate_id < 0:
            raise ValueError("candidate_id must be nonnegative")


def video_point(x: float, y: float) -> Point2D:
    return Point2D(x, y, CoordinateSpace.VIDEO)


__all__ = [
    "CandidateCacheFrameRef",
    "DecisionFrameRecord",
    "SliderPathCandidateRef",
    "SpatialCandidateRef",
    "TemporalTargetRef",
    "video_point",
]
