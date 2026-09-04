"""验证 Phase 1 的类型边界，而不是依赖“调用方自觉不读 GT”。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
from package import DataSplit as PackageDataSplit

from traning.state import (
    BeliefState,
    CandidateObservation,
    DataSplit,
    DecisionAction,
    DecisionResult,
    InferenceCandidateRecord,
    OutcomeDistribution,
    RuntimeFrame,
    TrainingSample,
    TrackedObservation,
)

_RUNTIME_CONTRACTS = (
    RuntimeFrame,
    InferenceCandidateRecord,
    CandidateObservation,
    TrackedObservation,
    BeliefState,
    OutcomeDistribution,
    DecisionResult,
)
_GT_ONLY_NAMES = frozenset(
    {
        "temporal_target",
        "selected_candidate_id",
        "hit_objects",
        "visible_hit_objects",
        "gt_timing",
        "gt_score",
        "oracle_label",
        "target_label",
    }
)


def test_data_split_has_one_canonical_vocabulary() -> None:
    """固定跨数据层唯一允许的 split 值。"""

    assert {item.name for item in DataSplit} == {
        "ALL",
        "TRAIN",
        "VALIDATION",
        "TEST",
    }
    assert {item.value for item in DataSplit} == {
        "all",
        "train",
        "validation",
        "test",
    }
    assert DataSplit is PackageDataSplit


@pytest.mark.parametrize("contract_type", _RUNTIME_CONTRACTS)
def test_runtime_contracts_structurally_exclude_gt(contract_type: type[object]) -> None:
    """从 dataclass schema 层阻止 runtime 获得任何 GT-only 字段。"""

    field_names = {item.name for item in fields(contract_type)}
    assert field_names.isdisjoint(_GT_ONLY_NAMES)
    assert hasattr(contract_type, "__slots__")
    assert contract_type.__dataclass_params__.frozen


def test_runtime_instance_cannot_gain_gt_attribute() -> None:
    """slots + frozen 使推理对象无法在运行时偷偷补入 GT。"""

    frame = RuntimeFrame(
        frame_id="frame-1",
        frame_index=0,
        timestamp_ms=0.0,
        width=1,
        height=1,
        image_bytes=b"\x00",
    )
    with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
        frame.temporal_target = {"action": "click"}  # type: ignore[attr-defined]


def test_training_sample_requires_coordinate_transform_fingerprint() -> None:
    """训练 target 必须显式绑定生成它的坐标变换。"""

    sample = TrainingSample(
        sample_id="sample-1",
        split=DataSplit.TRAIN,
        frame_index=0,
        timestamp_ms=0.0,
        width=1484,
        height=846,
        image_bytes=b"frame",
        transform_fingerprint="transform-0123456789abcdef",
        candidates=(),
        ground_truth_objects=(),
        selected_candidate_id=None,
    )
    assert sample.transform_fingerprint == "transform-0123456789abcdef"

    with pytest.raises(ValueError, match="transform_fingerprint"):
        TrainingSample(
            sample_id="sample-legacy",
            split=DataSplit.TRAIN,
            frame_index=0,
            timestamp_ms=0.0,
            width=1484,
            height=846,
            image_bytes=b"frame",
            transform_fingerprint="missing-coordinate-identity",
            candidates=(),
            ground_truth_objects=(),
            selected_candidate_id=None,
        )


def test_outcome_distribution_rejects_invalid_probability_mass() -> None:
    """Outcome 五类概率必须形成一个规范化离散分布。"""

    with pytest.raises(ValueError):
        OutcomeDistribution(
            track_id="track-1",
            horizon_ms=16.0,
            p_invalid=0.2,
            p_miss=0.2,
            p_low_score=0.2,
            p_medium_score=0.2,
            p_high_score=0.3,
            expected_score=0.5,
            variance=0.1,
            p_expire=0.0,
        )


def test_wait_decision_cannot_select_a_track() -> None:
    """WAIT 与 CLICK 的目标约束在 contract 构造时确定。"""

    with pytest.raises(ValueError):
        DecisionResult(
            action=DecisionAction.WAIT,
            track_id="track-7",
            execute_at_ms=1000.0,
            expected_utility=0.4,
            wait_utility=0.5,
            confidence=0.8,
        )


def test_phase1_core_does_not_import_typing_any() -> None:
    """在核心长期接口中禁止重新引入宽泛 Any。"""

    package_root = Path(__file__).resolve().parents[2]
    for directory_name in ("contracts", "config", "infrastructure"):
        for path in (package_root / directory_name).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "typing":
                    assert all(alias.name != "Any" for alias in node.names), path
                if isinstance(node, ast.Name):
                    assert node.id != "Any", path
