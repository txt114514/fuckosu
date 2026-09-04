"""验证训练包采用与 before_traning 对齐的显式分层。"""

from __future__ import annotations

import ast
from pathlib import Path


_TRAINING_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_PATHS = (
    "main.py",
    "__main__.py",
    "conf",
    "core",
    "core/app",
    "lib",
    "lib/environment",
    "lib/validation",
    "state",
    "tests",
    "docs",
)
_RETIRED_PREFIXES = (
    "traning.app",
    "traning.config",
    "traning.contracts",
    "traning.data",
    "traning.perception",
    "traning.tracking",
    "traning.belief",
    "traning.outcome",
    "traning.decision",
    "traning.evaluation",
    "traning.training",
    "traning.telemetry",
    "traning.visualization",
    "traning.infrastructure",
)


def _absolute_imports(path: Path) -> tuple[str, ...]:
    """返回源码中的绝对 import module 名。"""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return tuple(modules)


def test_required_layered_paths_exist() -> None:
    """硬标准列出的入口与权威目录必须全部存在。"""

    missing = tuple(
        relative
        for relative in _REQUIRED_PATHS
        if not (_TRAINING_ROOT / relative).exists()
    )
    assert not missing


def test_canonical_source_does_not_depend_on_flat_compatibility_packages() -> None:
    """新实现只能从 conf/core/lib/state 导入，不能反向依赖 wrapper。"""

    roots = tuple(
        _TRAINING_ROOT / name for name in ("conf", "core", "lib", "state")
    )
    violations: list[str] = []
    source_paths = sorted(path for root in roots for path in root.rglob("*.py"))
    for source_path in source_paths:
        for module in _absolute_imports(source_path):
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in _RETIRED_PREFIXES
            ):
                violations.append(
                    f"{source_path.relative_to(_TRAINING_ROOT)} -> {module}"
                )
    assert not violations, "canonical 源码仍依赖旧扁平入口：\n" + "\n".join(
        violations
    )


def test_before_traning_and_traning_share_explicit_entry_and_layers() -> None:
    """两个训练阶段包都必须具有入口、配置、核心、状态、测试和文档层。"""

    source_root = _TRAINING_ROOT.parent
    before = source_root / "before_traning"
    for package_root in (before, _TRAINING_ROOT):
        assert (package_root / "main.py").is_file()
        assert (package_root / "conf").is_dir()
        assert (package_root / "core").is_dir()
        assert (package_root / "state").is_dir()
        assert (package_root / "tests").is_dir()
        assert (package_root / "docs").is_dir()
