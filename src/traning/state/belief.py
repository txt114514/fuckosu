"""逐轨迹时序信念契约。"""

from dataclasses import dataclass

from .common import (
    require_finite,
    require_identifier,
    require_nonnegative,
    require_probability,
)
from .observation import ObjectTypeDistribution, Point2D


@dataclass(frozen=True, slots=True)
class BeliefState:
    """只由历史观测形成的因果轨迹信念。"""

    track_id: str
    timestamp_ms: float
    belief_embedding: tuple[float, ...]
    position_mean: Point2D
    position_uncertainty: Point2D
    visibility_probability: float
    object_type_distribution: ObjectTypeDistribution
    age: int
    time_since_seen_ms: float
    uncertainty: float

    def __post_init__(self) -> None:
        require_identifier(self.track_id, "track_id")
        require_nonnegative(self.timestamp_ms, "timestamp_ms")
        if not self.belief_embedding:
            raise ValueError("belief_embedding 不得为空")
        for index, value in enumerate(self.belief_embedding):
            require_finite(value, f"belief_embedding[{index}]")
        if not isinstance(self.position_mean, Point2D):
            raise TypeError("position_mean 必须是 Point2D")
        if not isinstance(self.position_uncertainty, Point2D):
            raise TypeError("position_uncertainty 必须是 Point2D")
        if self.position_uncertainty.x < 0.0 or self.position_uncertainty.y < 0.0:
            raise ValueError("position_uncertainty 的分量不得为负数")
        require_probability(self.visibility_probability, "visibility_probability")
        if not isinstance(self.object_type_distribution, ObjectTypeDistribution):
            raise TypeError("object_type_distribution 必须是 ObjectTypeDistribution")
        if isinstance(self.age, bool) or not isinstance(self.age, int):
            raise TypeError("age 必须是整数")
        if self.age < 1:
            raise ValueError("age 必须至少为 1")
        require_nonnegative(self.time_since_seen_ms, "time_since_seen_ms")
        require_nonnegative(self.uncertainty, "uncertainty")
