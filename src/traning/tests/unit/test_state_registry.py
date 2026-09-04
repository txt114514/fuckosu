"""统一 state 注册表覆盖硬标准类型且旧名称仅为 identity alias。"""

from __future__ import annotations

import pytest
from package import Box2D as PackageBox2D
from package import Point2D as PackagePoint2D
from package import Size2D as PackageSize2D

from traning import state


_EXPECTED_TYPES = {
    "ActionPrediction",
    "ActionType",
    "BeliefState",
    "Box2D",
    "Candidate",
    "CandidateBatch",
    "Circle2D",
    "EnvironmentReport",
    "FrameBatch",
    "LabelBatch",
    "LossBreakdown",
    "MemoryReport",
    "ObjectType",
    "OutcomePrediction",
    "PackageCheck",
    "Point2D",
    "ResizeMeta",
    "Size2D",
    "SpatialPrediction",
    "TorchCheck",
    "TrackState",
    "TrainingBatch",
    "TrainingSample",
    "VideoFrame",
}


def test_type_registry_has_exact_required_canonical_vocabulary() -> None:
    """注册表键与类型自身名称一致，避免字符串到错误类的隐式映射。"""

    assert set(state.TYPE_REGISTRY) == _EXPECTED_TYPES
    assert all(
        name == registered_type.__name__
        for name, registered_type in state.TYPE_REGISTRY.items()
    )
    assert len(set(state.TYPE_REGISTRY.values())) == len(state.TYPE_REGISTRY)


def test_legacy_type_names_are_identity_aliases_not_duplicate_definitions() -> None:
    """旧 import 保持工作，但运行时对象只有一份。"""

    aliases = {
        "CandidateObservation": (state.CandidateObservation, state.Candidate),
        "DecisionAction": (state.DecisionAction, state.ActionType),
        "DecisionResult": (state.DecisionResult, state.ActionPrediction),
        "OutcomeDistribution": (state.OutcomeDistribution, state.OutcomePrediction),
        "RuntimeFrame": (state.RuntimeFrame, state.VideoFrame),
        "TrackLifecycle": (state.TrackLifecycle, state.TrackState),
    }
    for name, (legacy_type, canonical_type) in aliases.items():
        assert legacy_type is canonical_type
        assert state.TYPE_ALIASES[name] is canonical_type
        assert state.registered_type(name) is canonical_type


def test_state_reuses_cross_module_geometry_authority() -> None:
    """共享几何留在 package，state 只注册同一个类型对象。"""

    assert state.Point2D is PackagePoint2D
    assert state.Size2D is PackageSize2D
    assert state.Box2D is PackageBox2D


def test_registry_is_read_only_and_strict_lookup_can_reject_aliases() -> None:
    """运行中不能静默替换全局类型权威。"""

    with pytest.raises(TypeError):
        state.TYPE_REGISTRY["VideoFrame"] = object  # type: ignore[index]
    with pytest.raises(KeyError):
        state.registered_type("RuntimeFrame", allow_alias=False)
