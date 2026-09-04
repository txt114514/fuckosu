"""感知观测与跟踪观测契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .common import (
    require_finite,
    require_identifier,
    require_nonnegative,
    require_probability,
    require_probability_sum,
)
from .geometry import Point2D


class ObjectType(str, Enum):
    """候选物体类型。"""

    RING = "ring"
    SLIDER = "slider"
    SPINNER = "spinner"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ObjectTypeDistribution:
    """候选物体类型的规范化概率分布。"""

    p_ring: float
    p_slider: float
    p_spinner: float
    p_unknown: float

    def __post_init__(self) -> None:
        require_probability_sum(
            (self.p_ring, self.p_slider, self.p_spinner, self.p_unknown),
            "object_type_distribution",
        )


@dataclass(frozen=True, slots=True)
class RingAttributes:
    """ring 分支能在 runtime 直接观测到的概率与像素半径。"""

    probability: float
    radius_px: float

    def __post_init__(self) -> None:
        require_probability(self.probability, "probability")
        require_nonnegative(self.radius_px, "radius_px")


@dataclass(frozen=True, slots=True)
class SliderAttributes:
    """slider 分支概率、局部方向和可选的已解码路径。"""

    probability: float
    direction: Point2D
    path: tuple[Point2D, ...] = ()

    def __post_init__(self) -> None:
        require_probability(self.probability, "probability")
        if len(self.path) == 1:
            raise ValueError("已解码 slider path 必须为空或至少包含两个点")


@dataclass(frozen=True, slots=True)
class SpinnerAttributes:
    """spinner 分支在当前帧的存在概率。"""

    probability: float

    def __post_init__(self) -> None:
        require_probability(self.probability, "probability")


@dataclass(frozen=True, slots=True)
class Candidate:
    """单帧感知产生的候选观测，不包含任何真值。"""

    frame_id: str
    frame_index: int
    timestamp_ms: float
    candidate_id: str
    x: float
    y: float
    confidence: float
    visibility_probability: float
    object_type_distribution: ObjectTypeDistribution
    appearance_embedding: tuple[float, ...]
    ring: RingAttributes | None = None
    slider: SliderAttributes | None = None
    spinner: SpinnerAttributes | None = None

    def __post_init__(self) -> None:
        require_identifier(self.frame_id, "frame_id")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index 必须是整数")
        if self.frame_index < 0:
            raise ValueError("frame_index 不得为负数")
        require_nonnegative(self.timestamp_ms, "timestamp_ms")
        require_identifier(self.candidate_id, "candidate_id")
        require_finite(self.x, "x")
        require_finite(self.y, "y")
        require_probability(self.confidence, "confidence")
        require_probability(self.visibility_probability, "visibility_probability")
        if not isinstance(self.object_type_distribution, ObjectTypeDistribution):
            raise TypeError("object_type_distribution 必须是 ObjectTypeDistribution")
        if not self.appearance_embedding:
            raise ValueError("appearance_embedding 不得为空")
        for index, value in enumerate(self.appearance_embedding):
            require_finite(value, f"appearance_embedding[{index}]")
        if not any(value != 0.0 for value in self.appearance_embedding):
            raise ValueError("appearance_embedding 不得是零向量")
        if (
            sum(value is not None for value in (self.ring, self.slider, self.spinner))
            > 1
        ):
            raise ValueError("ring、slider、spinner 专属属性最多只能存在一组")


class TrackState(str, Enum):
    """轨迹生命周期状态。"""

    NEW = "new"
    ACTIVE = "active"
    MISSING = "missing"
    EXPIRED = "expired"


class AssociationStatus(str, Enum):
    """当前帧的关联结果。"""

    CREATED = "created"
    MATCHED = "matched"
    UNMATCHED = "unmatched"


@dataclass(frozen=True, slots=True)
class TrackedObservation:
    """带稳定轨迹身份、生命周期和关联信息的观测。"""

    track_id: str
    frame_id: str
    frame_index: int
    timestamp_ms: float
    lifecycle: TrackState
    association: AssociationStatus
    association_confidence: float
    track_age: int
    missed_frames: int
    time_since_seen_ms: float
    candidate: Candidate | None
    association_cost: float | None = None

    def __post_init__(self) -> None:
        require_identifier(self.track_id, "track_id")
        require_identifier(self.frame_id, "frame_id")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index 必须是整数")
        if self.frame_index < 0:
            raise ValueError("frame_index 不得为负数")
        require_nonnegative(self.timestamp_ms, "timestamp_ms")
        if not isinstance(self.lifecycle, TrackState):
            raise TypeError("lifecycle 必须是 TrackState")
        if not isinstance(self.association, AssociationStatus):
            raise TypeError("association 必须是 AssociationStatus")
        require_probability(self.association_confidence, "association_confidence")
        if isinstance(self.track_age, bool) or not isinstance(self.track_age, int):
            raise TypeError("track_age 必须是整数")
        if self.track_age < 1:
            raise ValueError("track_age 必须至少为 1")
        if isinstance(self.missed_frames, bool) or not isinstance(
            self.missed_frames, int
        ):
            raise TypeError("missed_frames 必须是整数")
        if self.missed_frames < 0:
            raise ValueError("missed_frames 不得为负数")
        require_nonnegative(self.time_since_seen_ms, "time_since_seen_ms")
        if self.association_cost is not None:
            require_nonnegative(self.association_cost, "association_cost")
        if self.candidate is not None and not isinstance(self.candidate, Candidate):
            raise TypeError("candidate 必须是 Candidate（兼容名 CandidateObservation）")
        if self.association is AssociationStatus.CREATED:
            if self.lifecycle is not TrackState.NEW or self.candidate is None:
                raise ValueError("created 关联必须是携带 candidate 的 NEW 轨迹")
            if (
                self.track_age != 1
                or self.missed_frames != 0
                or self.time_since_seen_ms != 0.0
            ):
                raise ValueError("新轨迹必须从 age=1、missed=0、time_since_seen=0 开始")
            if self.association_cost is not None:
                raise ValueError("created 关联没有 association_cost")
        elif self.association is AssociationStatus.MATCHED:
            if self.lifecycle is not TrackState.ACTIVE or self.candidate is None:
                raise ValueError("matched 关联必须是携带 candidate 的 ACTIVE 轨迹")
            if self.missed_frames != 0 or self.time_since_seen_ms != 0.0:
                raise ValueError("匹配轨迹必须清零 missed/time_since_seen")
            if self.association_cost is None:
                raise ValueError("matched 关联必须携带 association_cost")
        else:
            if self.lifecycle not in (TrackState.MISSING, TrackState.EXPIRED):
                raise ValueError("unmatched 关联只能处于 MISSING 或 EXPIRED")
            if self.candidate is not None or self.association_confidence != 0.0:
                raise ValueError(
                    "unmatched 关联不得携带 candidate，confidence 必须为 0"
                )
            if self.missed_frames < 1:
                raise ValueError("unmatched 轨迹的 missed_frames 必须至少为 1")
            if self.association_cost is not None:
                raise ValueError("unmatched 关联没有 association_cost")
        if self.candidate is not None and (
            self.candidate.frame_id != self.frame_id
            or self.candidate.frame_index != self.frame_index
            or self.candidate.timestamp_ms != self.timestamp_ms
        ):
            raise ValueError("candidate 的帧身份必须与 TrackedObservation 一致")


# 旧名称只保留 identity alias，不复制 dataclass 或 Enum。
CandidateObservation = Candidate
TrackLifecycle = TrackState


__all__ = [
    "AssociationStatus",
    "Candidate",
    "CandidateObservation",
    "ObjectType",
    "ObjectTypeDistribution",
    "Point2D",
    "RingAttributes",
    "SliderAttributes",
    "SpinnerAttributes",
    "TrackLifecycle",
    "TrackState",
    "TrackedObservation",
]
