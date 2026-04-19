from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol


__all__ = [
    "BatchProcessResult",
    "ConfigValueSpec",
    "FailedReportStoreLike",
    "FolderBatchProcessor",
    "FolderWalkerLike",
    "config_filenames",
    "config_nonempty_strs",
    "config_nonnegative_ints",
    "config_positive_floats",
    "config_positive_ints",
    "config_resolved_paths",
    "config_string_tuples",
    "config_suffix_tuples",
    "merge_config_specs",
    "prefix_config_keys",
    "read_config_values",
]


BatchProcessResult = Literal["success", "skip"]


@dataclass(frozen=True)
class ConfigValueSpec:
    key: str
    reader_name: str
    paths: tuple[tuple[str, ...], ...]


ConfigPathInput = str | tuple[str, ...]
ConfigPathGroupInput = str | tuple[str, ...]


def _normalize_config_path(path: ConfigPathInput) -> tuple[str, ...]:
    if isinstance(path, str):
        normalized = tuple(part for part in path.split(".") if part)
    else:
        normalized = path

    if not normalized:
        raise ValueError("配置路径不能为空")
    return normalized


def _normalize_config_path_group(paths: ConfigPathGroupInput) -> tuple[tuple[str, ...], ...]:
    if isinstance(paths, str):
        return (_normalize_config_path(paths),)

    if not paths:
        raise ValueError("配置路径组不能为空")

    if all(isinstance(part, str) for part in paths):
        dotted_text = "." in paths[0]
        if dotted_text:
            return tuple(_normalize_config_path(path) for path in paths)
        return (_normalize_config_path(paths),)

    raise TypeError(f"不支持的配置路径定义: {paths!r}")


def _config_values(reader_name: str, **entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]:
    return tuple(
        ConfigValueSpec(
            key=key,
            reader_name=reader_name,
            paths=_normalize_config_path_group(paths),
        )
        for key, paths in entries.items()
    )


def config_resolved_paths(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]:
    return _config_values("resolved_path", **entries)


def config_filenames(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]:
    return _config_values("filename", **entries)


def config_nonempty_strs(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]:
    return _config_values("nonempty_str", **entries)


def config_string_tuples(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]:
    return _config_values("string_tuple", **entries)


def config_suffix_tuples(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]:
    return _config_values("suffix_tuple", **entries)


def config_positive_ints(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]:
    return _config_values("positive_int", **entries)


def config_nonnegative_ints(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]:
    return _config_values("nonnegative_int", **entries)


def config_positive_floats(**entries: ConfigPathGroupInput) -> tuple[ConfigValueSpec, ...]:
    return _config_values("positive_float", **entries)


def merge_config_specs(
    *spec_groups: ConfigValueSpec | Iterable[ConfigValueSpec],
) -> tuple[ConfigValueSpec, ...]:
    merged: list[ConfigValueSpec] = []
    for group in spec_groups:
        if isinstance(group, ConfigValueSpec):
            merged.append(group)
        else:
            merged.extend(group)
    return tuple(merged)


def prefix_config_keys(
    specs: Iterable[ConfigValueSpec],
    prefix: str,
) -> tuple[ConfigValueSpec, ...]:
    return tuple(
        ConfigValueSpec(
            key=f"{prefix}{spec.key}",
            reader_name=spec.reader_name,
            paths=spec.paths,
        )
        for spec in specs
    )


def read_config_values(
    config_reader: Any,
    *spec_groups: ConfigValueSpec | Iterable[ConfigValueSpec],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for group in spec_groups:
        if isinstance(group, ConfigValueSpec):
            specs = (group,)
        else:
            specs = group

        for spec in specs:
            value = config_reader.read(spec.reader_name, *spec.paths)
            if value is not None:
                resolved[spec.key] = value
    return resolved


class FolderWalkerLike(Protocol):
    def read_folder_names(self) -> list[str]:
        ...


class FailedReportStoreLike(Protocol):
    def write_failed_report(
        self,
        failed_cases: list[tuple[str, str]],
        failed_filename: str,
    ) -> Path:
        ...


class FolderBatchProcessor(ABC):
    """Shared shell for folder-based batch processors."""

    walker: FolderWalkerLike
    store: FailedReportStoreLike
    failed_filename: str

    def __init__(self, failed_filename: str):
        self.failed_filename = failed_filename
        self.success_count = 0
        self.skip_count = 0
        self.fail_count = 0
        self.failed_cases: list[tuple[str, str]] = []

    def progress_message(
        self,
        index: int,
        total: int,
        folder_name: str,
    ) -> str | None:
        return None

    def iter_folder_names(self) -> list[str]:
        return self.walker.read_folder_names()

    def handle_failure(self, folder_name: str, error: Exception):
        pass

    @abstractmethod
    def process_one(
        self,
        folder_name: str,
        overwrite: bool = False,
    ) -> BatchProcessResult:
        raise NotImplementedError

    def _record_result(self, folder_name: str, result: BatchProcessResult):
        if result == "success":
            self.success_count += 1
            print(f"[完成] {folder_name}")
            return

        if result == "skip":
            self.skip_count += 1
            print(f"[跳过] {folder_name}")
            return

        raise ValueError(f"未知处理结果: {result}")

    def _print_summary(self, failed_path: Path):
        print()
        print(
            f"处理完成：成功 {self.success_count} 个，跳过 {self.skip_count} 个，失败 {self.fail_count} 个"
        )
        print(f"失败名单：{failed_path}")

    def run(self, overwrite: bool = False):
        folder_names = self.iter_folder_names()
        total = len(folder_names)

        for index, folder_name in enumerate(folder_names, start=1):
            progress = self.progress_message(index, total, folder_name)
            if progress is not None:
                print(progress)

            try:
                result = self.process_one(folder_name, overwrite=overwrite)
            except Exception as e:
                self.fail_count += 1
                self.failed_cases.append((folder_name, str(e)))
                self.handle_failure(folder_name, e)
                print(f"[失败] {folder_name}: {e}")
                continue

            self._record_result(folder_name, result)

        failed_path = self.store.write_failed_report(
            self.failed_cases,
            failed_filename=self.failed_filename,
        )
        self._print_summary(failed_path)
