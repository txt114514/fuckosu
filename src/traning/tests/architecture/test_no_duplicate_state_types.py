"""验证硬标准核心类型只有一个权威定义与对象 identity。"""

from __future__ import annotations

import ast
from pathlib import Path

from traning.state import TYPE_REGISTRY


_SRC_ROOT = Path(__file__).resolve().parents[3]
_REQUIRED_NAMES = {
    "ResizeMeta",
    "Point2D",
    "Size2D",
    "Box2D",
    "Circle2D",
    "VideoFrame",
    "FrameBatch",
    "LabelBatch",
    "TrainingSample",
    "TrainingBatch",
    "SpatialPrediction",
    "Candidate",
    "CandidateBatch",
    "ObjectType",
    "TrackState",
    "BeliefState",
    "OutcomePrediction",
    "ActionType",
    "ActionPrediction",
    "LossBreakdown",
    "MemoryReport",
    "EnvironmentReport",
    "PackageCheck",
    "TorchCheck",
}
_WRAPPER_PARTS = {
    "app",
    "belief",
    "config",
    "contracts",
    "data",
    "decision",
    "evaluation",
    "infrastructure",
    "outcome",
    "perception",
    "telemetry",
    "tracking",
    "training",
    "visualization",
}


def test_registry_contains_every_required_type_once() -> None:
    """核心词汇必须完整、只读并映射到互不重复的 type 对象。"""

    assert set(TYPE_REGISTRY) == _REQUIRED_NAMES
    assert len(set(TYPE_REGISTRY.values())) == len(_REQUIRED_NAMES)
    assert all(isinstance(value, type) for value in TYPE_REGISTRY.values())


def test_required_names_have_at_most_one_canonical_class_definition() -> None:
    """alias 可以没有同名 class，但任何规范名称不能出现两个 class 定义。"""

    definitions: dict[str, list[str]] = {name: [] for name in _REQUIRED_NAMES}
    roots = (_SRC_ROOT / "traning", _SRC_ROOT / "package", _SRC_ROOT / "start")
    for root in roots:
        for path in root.rglob("*.py"):
            relative = path.relative_to(_SRC_ROOT)
            if "tests" in relative.parts or "__pycache__" in relative.parts:
                continue
            if relative.parts[:1] == ("traning",) and len(relative.parts) > 1:
                if relative.parts[1] in _WRAPPER_PARTS:
                    continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name in definitions:
                    definitions[node.name].append(str(relative))
    duplicates = {
        name: paths for name, paths in definitions.items() if len(paths) > 1
    }
    assert not duplicates


def test_registry_types_come_from_state_or_shared_package() -> None:
    """注册对象的 module 必须指向 state 或跨模块共享 package。"""

    invalid = {
        name: registered.__module__
        for name, registered in TYPE_REGISTRY.items()
        if not (
            registered.__module__.startswith("traning.state")
            or registered.__module__.startswith("package.contracts")
        )
    }
    assert not invalid
