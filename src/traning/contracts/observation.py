"""已弃用兼容转发；新代码必须导入对应的 conf、core、lib 或 state 路径。"""

from traning.state.observation import (
    AssociationStatus,
    CandidateObservation,
    ObjectType,
    ObjectTypeDistribution,
    Point2D,
    RingAttributes,
    SliderAttributes,
    SpinnerAttributes,
    TrackedObservation,
    TrackLifecycle,
)

__deprecated__ = True
__all__ = [
    "AssociationStatus",
    "CandidateObservation",
    "ObjectType",
    "ObjectTypeDistribution",
    "Point2D",
    "RingAttributes",
    "SliderAttributes",
    "SpinnerAttributes",
    "TrackedObservation",
    "TrackLifecycle",
]
