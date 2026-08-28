"""Phase 11 生产源码注释完整性与 legacy 退役边界验收。"""

from __future__ import annotations

import ast
import re
from pathlib import Path


_V2_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _V2_ROOT.parents[1]
_DOCSTRING_EXCLUSIONS: frozenset[Path] = frozenset()
_CHINESE_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_RUNTIME_FORBIDDEN_NAMES = frozenset(
    {
        "GT",
        "GroundTruthObject",
        "OutcomeOracle",
        "OutcomeTrainingSample",
        "OracleOutcome",
        "OracleState",
        "TrainingCandidateRecord",
        "TrainingSample",
        "action_logits",
        "candidate_logits",
        "ground_truth",
        "gt_score",
        "gt_timing",
        "hit_objects",
        "oracle",
        "oracle_label",
        "selected_candidate_id",
        "temporal_target",
    }
)


def _python_paths(*, include_tests: bool) -> tuple[Path, ...]:
    """稳定枚举 V2 Python 源码，并排除生成缓存与冻结清单。"""

    paths = []
    for path in _V2_ROOT.rglob("*.py"):
        relative = path.relative_to(_V2_ROOT)
        if "__pycache__" in relative.parts or "legacy" in relative.parts:
            continue
        if not include_tests and "tests" in relative.parts:
            continue
        paths.append(path)
    return tuple(sorted(paths))


def _tree(path: Path) -> ast.Module:
    """用带文件名的 UTF-8 AST 解析提供可定位的失败信息。"""

    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_names(tree: ast.AST) -> tuple[tuple[str, str], ...]:
    """返回 ``(module, imported_name)``，统一处理两种 import 语法。"""

    imports: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                (alias.name, alias.name.rsplit(".", 1)[-1]) for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.extend((module, alias.name) for alias in node.names)
    return tuple(imports)


def _is_runtime_or_decision(relative: Path) -> bool:
    """识别正式 runtime 与整个 deterministic decision 包。"""

    return (
        relative.parts[0] in {"belief", "decision", "tracking"}
        or relative.parts[:2]
        in {
            ("perception", "decode"),
            ("perception", "models"),
            ("perception", "runtime"),
        }
        or relative == Path("outcome/model.py")
        or relative.name == "runtime.py"
        or "runtime" in relative.parts
    )


def _chinese_docstring_violations(paths: tuple[Path, ...]) -> tuple[str, ...]:
    """返回指定源码中缺少中文模块或公开定义说明的位置。"""

    missing: list[str] = []
    for path in paths:
        relative = path.relative_to(_V2_ROOT)
        if relative in _DOCSTRING_EXCLUSIONS:
            continue
        tree = _tree(path)
        module_doc = ast.get_docstring(tree, clean=False)
        if module_doc is None or _CHINESE_TEXT.search(module_doc) is None:
            missing.append(f"{relative}:1 module")
        # ast.walk 也覆盖公开 nested writer；不能因它不是模块入口就省略契约说明。
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ) or node.name.startswith("_"):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if docstring is None or _CHINESE_TEXT.search(docstring) is None:
                kind = "class" if isinstance(node, ast.ClassDef) else "callable"
                missing.append(f"{relative}:{node.lineno} {kind} {node.name}")
    return tuple(missing)


def test_production_modules_and_public_definitions_have_chinese_docstrings() -> None:
    """生产模块及所有公开定义必须同时有说明和中文领域语义。"""

    missing = _chinese_docstring_violations(_python_paths(include_tests=False))
    assert not missing, "缺少中文 docstring：\n" + "\n".join(missing)


def test_test_modules_and_public_definitions_have_chinese_docstrings() -> None:
    """测试模块与公开定义也必须用中文说明其验证意图。"""

    test_paths = tuple(
        path
        for path in _python_paths(include_tests=True)
        if "tests" in path.relative_to(_V2_ROOT).parts
    )
    missing = _chinese_docstring_violations(test_paths)
    assert not missing, "缺少中文 docstring：\n" + "\n".join(missing)


def test_repository_python_modules_have_chinese_docstrings() -> None:
    """全仓 Python 模块必须说明用途；冻结 legacy 不做机械符号级改写。"""

    roots = tuple(
        path
        for path in (
            _WORKSPACE_ROOT / "src",
            _WORKSPACE_ROOT / "environment",
            _WORKSPACE_ROOT / "project_index",
        )
        if path.exists()
    )
    paths = tuple(
        sorted(
            {
                path
                for root in roots
                for path in root.rglob("*.py")
                if "__pycache__" not in path.parts
            }
            | set(_WORKSPACE_ROOT.glob("*.py"))
        )
    )
    missing: list[str] = []
    for path in paths:
        module_doc = ast.get_docstring(_tree(path), clean=False)
        if module_doc is None or _CHINESE_TEXT.search(module_doc) is None:
            missing.append(str(path.relative_to(_WORKSPACE_ROOT)))
    assert not missing, "全仓模块缺少中文用途说明：\n" + "\n".join(missing)


def test_production_has_no_retired_namespace_sparse_or_typing_any_dependency() -> None:
    """正式生产路径不得接回旧命名空间、稀疏主线或宽泛 Any。"""

    violations: list[str] = []
    for path in _python_paths(include_tests=False):
        relative = path.relative_to(_V2_ROOT)
        tree = _tree(path)
        for module, imported_name in _imported_names(tree):
            normalized = f"{module}.{imported_name}".lower()
            if module == "osu_v2" or module.startswith("osu_v2."):
                violations.append(f"{relative}: retired namespace import {module}")
            if "smet" in normalized or "dynamicsparse" in normalized:
                violations.append(f"{relative}: sparse import {normalized}")
            if module == "typing" and imported_name == "Any":
                violations.append(f"{relative}: typing.Any")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "Any":
                violations.append(f"{relative}:{node.lineno} Any")
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "typing"
                and node.attr == "Any"
            ):
                violations.append(f"{relative}:{node.lineno} typing.Any")
    assert not violations, "V2 生产依赖越界：\n" + "\n".join(violations)


def test_runtime_and_decision_cannot_import_training_or_oracle_information() -> None:
    """正式动作路径只允许消费 runtime contracts，不得获得 GT/oracle/logits。"""

    violations: list[str] = []
    for path in _python_paths(include_tests=False):
        relative = path.relative_to(_V2_ROOT)
        if not _is_runtime_or_decision(relative):
            continue
        tree = _tree(path)
        for module, imported_name in _imported_names(tree):
            if module.startswith("traning.outcome.oracle"):
                violations.append(f"{relative}: oracle import {module}")
            if imported_name in _RUNTIME_FORBIDDEN_NAMES:
                violations.append(f"{relative}: forbidden import {imported_name}")
        for node in ast.walk(tree):
            name = ""
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            if name in _RUNTIME_FORBIDDEN_NAMES:
                violations.append(f"{relative}:{node.lineno} forbidden name {name}")
    assert not violations, "runtime/decision 信息泄漏：\n" + "\n".join(violations)


def test_tests_do_not_import_retired_parallel_namespace() -> None:
    """迁入后的测试也不得重新依赖已删除的并行包。"""

    violations: list[str] = []
    for path in _python_paths(include_tests=True):
        relative = path.relative_to(_V2_ROOT)
        imports_retired_namespace = any(
            module == "osu_v2" or module.startswith("osu_v2.")
            for module, _name in _imported_names(_tree(path))
        )
        if imports_retired_namespace:
            violations.append(str(relative))
    assert not violations, "测试仍导入已删除的并行命名空间：\n" + "\n".join(
        violations
    )
