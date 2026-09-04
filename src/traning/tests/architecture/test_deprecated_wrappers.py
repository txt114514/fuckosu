"""验证旧扁平入口只是带弃用标记的 identity 转发。"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from traning import app as legacy_app
from traning import config as legacy_config
from traning import contracts as legacy_contracts
from traning.conf import V2Config
from traning.core.app import V2RuntimePipeline
from traning.lib.environment import EnvironmentReport
from traning.state import TrainingSample


_WORKSPACE = Path(__file__).resolve().parents[4]
_TRAINING_ROOT = _WORKSPACE / "src" / "traning"
_WRAPPER_ROOTS = (
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
)


def test_minimum_legacy_packages_reexport_canonical_objects() -> None:
    """app、config、contracts 与根 environment 不得制造第二份对象。"""

    wrapper_path = _WORKSPACE / "environment" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "_deprecated_environment_wrapper",
        wrapper_path,
    )
    assert spec is not None and spec.loader is not None
    legacy_environment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_environment)

    assert legacy_app.V2RuntimePipeline is V2RuntimePipeline
    assert legacy_config.V2Config is V2Config
    assert legacy_contracts.TrainingSample is TrainingSample
    assert legacy_environment.EnvironmentReport is EnvironmentReport


def test_wrapper_modules_define_no_business_class_or_function() -> None:
    """所有物理迁移后的旧 Python 文件只能导入和转发。"""

    violations: list[str] = []
    for root_name in _WRAPPER_ROOTS:
        for path in (_TRAINING_ROOT / root_name).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            definitions = tuple(
                node.name
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            )
            if definitions:
                violations.append(
                    f"{path.relative_to(_TRAINING_ROOT)}: {definitions}"
                )
            module_doc = ast.get_docstring(tree, clean=False) or ""
            if "弃用" not in module_doc and "Deprecated" not in module_doc:
                violations.append(
                    f"{path.relative_to(_TRAINING_ROOT)}: 缺少 deprecated 模块说明"
                )
    assert not violations, "旧路径仍含实现：\n" + "\n".join(violations)


def test_root_environment_python_files_are_forwarders() -> None:
    """仓库根环境包不能继续包含 dataclass 或探测函数实现。"""

    for relative in ("environment/__init__.py", "environment/env_check.py"):
        path = _WORKSPACE / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            for node in tree.body
        )
