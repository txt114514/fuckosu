"""训练与推理跨层数据类型的唯一名称注册表。"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from .belief import BeliefState
from .data import FrameBatch, LabelBatch, TrainingBatch, TrainingSample, VideoFrame
from .decision import ActionPrediction, ActionType
from .environment import EnvironmentReport, PackageCheck, TorchCheck
from .geometry import Box2D, Circle2D, Point2D, ResizeMeta, Size2D
from .observation import Candidate, ObjectType, TrackState
from .outcome import OutcomePrediction
from .perception import CandidateBatch, SpatialPrediction
from .telemetry import MemoryReport
from .training import LossBreakdown


_CORE_TYPES: tuple[type[object], ...] = (
    ResizeMeta,
    Point2D,
    Size2D,
    Box2D,
    Circle2D,
    VideoFrame,
    FrameBatch,
    LabelBatch,
    TrainingSample,
    TrainingBatch,
    SpatialPrediction,
    Candidate,
    CandidateBatch,
    ObjectType,
    TrackState,
    BeliefState,
    OutcomePrediction,
    ActionType,
    ActionPrediction,
    LossBreakdown,
    MemoryReport,
    EnvironmentReport,
    PackageCheck,
    TorchCheck,
)

TYPE_REGISTRY: Final = MappingProxyType(
    {registered_type.__name__: registered_type for registered_type in _CORE_TYPES}
)
"""规范类型名到类型对象的只读映射。"""

TYPE_ALIASES: Final = MappingProxyType(
    {
        "CandidateObservation": Candidate,
        "DecisionAction": ActionType,
        "DecisionResult": ActionPrediction,
        "MemorySnapshot": MemoryReport,
        "OutcomeDistribution": OutcomePrediction,
        "RuntimeFrame": VideoFrame,
        "TrackLifecycle": TrackState,
    }
)
"""旧名称到规范类型对象的只读兼容映射；不注册第二份定义。"""


def registered_type(name: str, *, allow_alias: bool = True) -> type[object]:
    """按稳定名称取得类型；未知名称以 ``KeyError`` 明确失败。"""

    try:
        return TYPE_REGISTRY[name]
    except KeyError:
        if allow_alias:
            return TYPE_ALIASES[name]
        raise


__all__ = ["TYPE_ALIASES", "TYPE_REGISTRY", "registered_type"]
