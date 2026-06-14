from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import pathspec

from before_traning.Lib.common.failures import exception_detail
from before_traning.Lib.common.pathspec import filter_files


ProcessResult = Literal["success", "skip"]
SortKey = Callable[[Path], object]
class FolderStoreLike(Protocol):
    def folder_exists(self, folder_name: str) -> bool:
        ...

    def file_exists(self, folder_name: str, filename: str) -> bool:
        ...

    def find_files(self, folder_name: str, pattern: str = "*") -> list[Path]:
        ...


class StatusManagerLike(Protocol):
    process_steps: Iterable[str]

    def ensure_status_file(self, folder_name: str) -> dict[str, Any]:
        ...

    def is_step_done(self, folder_name: str, step: str) -> bool:
        ...

    def mark_step_done(
        self,
        folder_name: str,
        step: str,
        detail: Any = None,
    ) -> None:
        ...

    def mark_step_pending(
        self,
        folder_name: str,
        step: str,
        detail: Any = None,
    ) -> None:
        ...


def matching_files(
    directory: Path,
    spec: pathspec.PathSpec,
    *,
    sort_key: SortKey | None = None,
) -> list[Path]:
    if not directory.is_dir():
        return []
    files = filter_files(directory.iterdir(), spec)
    return sorted(files, key=sort_key or (lambda path: path.name.lower()))


@dataclass(frozen=True)
class ProcessingGuard:
    store: FolderStoreLike
    status_manager: StatusManagerLike
    status_step: str
    required_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        registered = set(self.status_manager.process_steps)
        requested = {*self.required_steps, self.status_step}
        unknown = requested - registered
        if unknown:
            raise ValueError(
                f"状态步骤未注册: {', '.join(sorted(unknown))}"
            )

    def prepare_folder(
        self,
        folder_name: str,
        *,
        required_patterns: Iterable[str] = (),
    ) -> dict[str, tuple[Path, ...]] | None:
        if not self.store.folder_exists(folder_name):
            return None
        self.status_manager.ensure_status_file(folder_name)
        self.ensure_required_steps(folder_name)

        matched: dict[str, tuple[Path, ...]] = {}
        for pattern in required_patterns:
            files = tuple(self.store.find_files(folder_name, pattern))
            if not files:
                return None
            matched[pattern] = files
        return matched

    def ensure_required_steps(self, folder_name: str) -> None:
        missing = [
            step
            for step in self.required_steps
            if not self.status_manager.is_step_done(folder_name, step)
        ]
        if missing:
            raise RuntimeError(
                f"{folder_name} 缺少前置步骤: {', '.join(missing)}"
            )

    def output_files_exist(
        self,
        folder_name: str,
        filenames: Iterable[str],
    ) -> bool:
        return all(
            self.store.file_exists(folder_name, filename)
            for filename in filenames
        )

    def is_complete(
        self,
        folder_name: str,
        *,
        overwrite: bool,
        artifact_exists: bool = True,
        output_files: Iterable[str] = (),
    ) -> bool:
        if overwrite or not artifact_exists:
            return False
        filenames = tuple(output_files)
        if filenames and not self.output_files_exist(folder_name, filenames):
            return False
        return self.status_manager.is_step_done(
            folder_name,
            self.status_step,
        )

    def step_done(self, folder_name: str) -> bool:
        return self.status_manager.is_step_done(
            folder_name,
            self.status_step,
        )

    def reconcile_existing(
        self,
        folder_name: str,
        *,
        overwrite: bool,
        artifact_exists: bool,
        detail: Any = None,
    ) -> ProcessResult | None:
        if overwrite or not artifact_exists:
            return None
        was_done = self.step_done(folder_name)
        self.mark_done(folder_name, detail=detail)
        return "skip" if was_done else "success"

    def mark_done(self, folder_name: str, detail: Any = None) -> None:
        self.status_manager.mark_step_done(
            folder_name,
            self.status_step,
            detail=detail,
        )

    def record_failure(self, folder_name: str, error: Exception) -> None:
        if not self.store.folder_exists(folder_name):
            return
        self.status_manager.ensure_status_file(folder_name)
        self.status_manager.mark_step_pending(
            folder_name,
            self.status_step,
            detail=exception_detail(error),
        )


__all__ = [
    "FolderStoreLike",
    "ProcessResult",
    "ProcessingGuard",
    "StatusManagerLike",
    "matching_files",
]
