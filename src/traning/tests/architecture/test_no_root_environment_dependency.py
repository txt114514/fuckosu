"""确保活动源码不再依赖仓库根 environment 包。"""

from __future__ import annotations

import ast
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parents[3]


def _imports_root_environment(path: Path) -> bool:
    """判断模块是否从旧根包导入 environment。"""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "environment" or alias.name.startswith("environment.")
            for alias in node.names
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and (
                node.module == "environment"
                or node.module.startswith("environment.")
            )
        ):
            return True
    return False


def test_start_and_canonical_training_source_do_not_import_root_environment() -> None:
    """start 与训练权威实现必须从 traning.lib.environment 导入。"""

    roots = (
        _SRC_ROOT / "start",
        _SRC_ROOT / "traning" / "conf",
        _SRC_ROOT / "traning" / "core",
        _SRC_ROOT / "traning" / "lib",
        _SRC_ROOT / "traning" / "state",
    )
    violations = tuple(
        str(path.relative_to(_SRC_ROOT))
        for root in roots
        for path in root.rglob("*.py")
        if "tests" not in path.parts and _imports_root_environment(path)
    )
    assert not violations


def test_start_registry_names_the_canonical_environment_module() -> None:
    """启动注册源码应显式记录新环境权威路径。"""

    registry = (_SRC_ROOT / "start" / "checks" / "registry.py").read_text(
        encoding="utf-8"
    )
    assert "from traning.lib.environment import" in registry
    assert "from environment" not in registry
