from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceModuleEntry:
    key: str
    import_name: str
    source_dir: Path
    description: str
    public_entry: str
    cli_entry: str | None = None
    docs: tuple[Path, ...] = ()

    @property
    def importable(self) -> bool:
        return importlib.util.find_spec(self.import_name) is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "import_name": self.import_name,
            "source_dir": str(self.source_dir),
            "description": self.description,
            "public_entry": self.public_entry,
            "cli_entry": self.cli_entry,
            "docs": tuple(str(path) for path in self.docs),
            "importable": self.importable,
        }


START_ENTRY = SourceModuleEntry(
    key="start",
    import_name="start",
    source_dir=SRC_ROOT / "start",
    description="统一 src 启动入口、模块入口登记和启动前自检。",
    public_entry="start.main:app",
    cli_entry="python -m start",
    docs=(SRC_ROOT / "start" / "README.md",),
)


SOURCE_MODULE_ENTRIES: tuple[SourceModuleEntry, ...] = (
    SourceModuleEntry(
        key="package",
        import_name="package",
        source_dir=SRC_ROOT / "package",
        description="跨 src 顶层模块共享的稳定 API 与结构契约。",
        public_entry="package.__init__",
        docs=(SRC_ROOT / "package" / "README.md",),
    ),
    SourceModuleEntry(
        key="before_traning",
        import_name="before_traning",
        source_dir=SRC_ROOT / "before_traning",
        description="训练前数据准备、谱面/视频处理和片段导出。",
        public_entry="before_traning.main",
        cli_entry="PYTHONPATH=src python -m before_traning.main",
        docs=(
            SRC_ROOT / "before_traning" / "docs" / "README.md",
            SRC_ROOT / "before_traning" / "docs" / "CODEX_INDEX.md",
        ),
    ),
    SourceModuleEntry(
        key="traning",
        import_name="traning",
        source_dir=SRC_ROOT / "traning",
        description="模型训练、候选缓存、时序决策、评分和导出。",
        public_entry="traning.cli",
        cli_entry="PYTHONPATH=src python -m traning.cli",
        docs=(
            SRC_ROOT / "traning" / "docs" / "TRAINING_PLAN.md",
            SRC_ROOT / "traning" / "docs" / "CODEX_INDEX.md",
        ),
    ),
)


ALL_START_ENTRIES: tuple[SourceModuleEntry, ...] = (
    START_ENTRY,
    *SOURCE_MODULE_ENTRIES,
)


def source_module_entries(
    *,
    include_start: bool = False,
) -> tuple[SourceModuleEntry, ...]:
    return ALL_START_ENTRIES if include_start else SOURCE_MODULE_ENTRIES


def source_module_entry(key: str) -> SourceModuleEntry:
    for entry in ALL_START_ENTRIES:
        if entry.key == key:
            return entry
    raise KeyError(f"unknown src module entry: {key}")


__all__ = [
    "ALL_START_ENTRIES",
    "SOURCE_MODULE_ENTRIES",
    "SRC_ROOT",
    "START_ENTRY",
    "SourceModuleEntry",
    "source_module_entries",
    "source_module_entry",
]
