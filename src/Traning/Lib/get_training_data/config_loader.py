from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

from Traning.Lib.function_tools.functions_process_tool import (
    config_filenames,
    config_nonempty_strs,
    config_nonnegative_ints,
    config_positive_floats,
    config_positive_ints,
    config_resolved_paths,
    config_string_tuples,
    config_suffix_tuples,
    merge_config_specs,
    prefix_config_keys,
)


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
T = TypeVar("T")


class CheckDataConfigError(Exception):
    pass


TARGET_ROOT_CONFIG_SPECS = config_resolved_paths(
    target_root=("file_management.target_root", "check_data.target_root"),
)
EXPORT_DIR_CONFIG_SPECS = config_resolved_paths(
    export_dir=("file_management.export_dir", "check_data.export_dir"),
)
VIDEO_ROOT_CONFIG_SPECS = config_resolved_paths(
    video_root=("file_management.video_root", "video_match.video_root"),
)
KEYWORD_CONFIG_SPECS = config_nonempty_strs(
    keyword=("file_formats.keyword", "check_data.keyword"),
)
ORDER_FILENAME_CONFIG_SPECS = config_filenames(
    order_filename=("file_management.order_filename", "get_check_data.order_filename"),
)
VERIFY_FILENAME_CONFIG_SPECS = config_filenames(
    verify_filename=("file_management.verify_filename", "get_check_data.verify_filename"),
)
DIFFICULTY_FILENAME_CONFIG_SPECS = config_filenames(
    difficulty_filename=("file_management.difficulty_filename", "get_check_data.difficulty_filename"),
)
VERIFY_FAILED_FILENAME_CONFIG_SPECS = config_filenames(
    verify_failed_filename=(
        "file_management.verify_failed_filename",
        "get_check_data.verify_failed_filename",
    ),
)
DIFFICULTY_FAILED_FILENAME_CONFIG_SPECS = config_filenames(
    difficulty_failed_filename=(
        "file_management.difficulty_failed_filename",
        "get_check_data.difficulty_failed_filename",
    ),
)
AUDIO_FILENAME_CONFIG_SPECS = config_filenames(
    audio_filename=("file_management.audio_filename", "video_shared.audio_filename"),
)
OUTPUT_FILENAME_CONFIG_SPECS = config_filenames(
    output_filename=("file_management.output_filename", "video_shared.output_filename"),
)
VIDEO_SUFFIXES_CONFIG_SPECS = config_suffix_tuples(
    video_suffixes=("file_formats.video_suffixes", "video_shared.video_suffixes"),
)
AV_CONFIG_SPECS = (
    merge_config_specs(
        config_filenames(
            failed_filename=(
                "file_management.av_correspondence_failed_filename",
                "av_correspondence.failed_filename",
            ),
        ),
        config_nonempty_strs(
            status_step=("progress.status_steps.av_correspondence", "video_shared.status_step"),
        ),
        config_string_tuples(
            required_steps=(
                "progress.required_steps.av_correspondence",
                "av_correspondence.required_steps",
            ),
        ),
        config_positive_ints(
            sample_rate=(
                "parameters.av_correspondence.sample_rate",
                "av_correspondence.sample_rate",
            ),
            envelope_hz=(
                "parameters.av_correspondence.envelope_hz",
                "av_correspondence.envelope_hz",
            ),
            refine_hz=("parameters.av_correspondence.refine_hz", "av_correspondence.refine_hz"),
        ),
        config_positive_floats(
            refine_search_seconds=(
                "parameters.av_correspondence.refine_search_seconds",
                "av_correspondence.refine_search_seconds",
            ),
        ),
    )
)
CLIP_CONFIG_SPECS = (
    merge_config_specs(
        config_filenames(
            failed_filename=("file_management.clip_failed_filename", "clip.failed_filename"),
        ),
        config_nonempty_strs(
            status_step=("progress.status_steps.clip", "clip.status_step"),
        ),
        config_string_tuples(
            required_steps=("progress.required_steps.clip", "clip.required_steps"),
        ),
        config_positive_ints(
            crop_reference_width=(
                "parameters.clip.crop_reference_width",
                "clip.crop_reference_width",
            ),
            crop_reference_height=(
                "parameters.clip.crop_reference_height",
                "clip.crop_reference_height",
            ),
        ),
        config_nonnegative_ints(
            crop_left=("parameters.clip.crop_left", "clip.crop_left"),
            crop_top=("parameters.clip.crop_top", "clip.crop_top"),
            crop_right=("parameters.clip.crop_right", "clip.crop_right"),
            crop_bottom=("parameters.clip.crop_bottom", "clip.crop_bottom"),
        ),
    )
)
PREFIXED_CLIP_CONFIG_SPECS = prefix_config_keys(CLIP_CONFIG_SPECS, "clip_")

CHECK_DATA_PIPELINE_CONFIG_SPECS = merge_config_specs(
    EXPORT_DIR_CONFIG_SPECS,
    TARGET_ROOT_CONFIG_SPECS,
    KEYWORD_CONFIG_SPECS,
    ORDER_FILENAME_CONFIG_SPECS,
    AUDIO_FILENAME_CONFIG_SPECS,
    VERIFY_FILENAME_CONFIG_SPECS,
    DIFFICULTY_FILENAME_CONFIG_SPECS,
    VERIFY_FAILED_FILENAME_CONFIG_SPECS,
    DIFFICULTY_FAILED_FILENAME_CONFIG_SPECS,
)
OSU_OSZ_PROCESSOR_CONFIG_SPECS = merge_config_specs(
    EXPORT_DIR_CONFIG_SPECS,
    TARGET_ROOT_CONFIG_SPECS,
    KEYWORD_CONFIG_SPECS,
    ORDER_FILENAME_CONFIG_SPECS,
    AUDIO_FILENAME_CONFIG_SPECS,
)
VERIFY_EXPORTER_CONFIG_SPECS = merge_config_specs(
    TARGET_ROOT_CONFIG_SPECS,
    ORDER_FILENAME_CONFIG_SPECS,
    VERIFY_FILENAME_CONFIG_SPECS,
    VERIFY_FAILED_FILENAME_CONFIG_SPECS,
)
VIDEO_PACKAGE_RENAMER_CONFIG_SPECS = merge_config_specs(
    VIDEO_ROOT_CONFIG_SPECS,
    TARGET_ROOT_CONFIG_SPECS,
    ORDER_FILENAME_CONFIG_SPECS,
    VIDEO_SUFFIXES_CONFIG_SPECS,
)
AV_CORRESPONDENCE_PROCESSOR_CONFIG_SPECS = merge_config_specs(
    TARGET_ROOT_CONFIG_SPECS,
    ORDER_FILENAME_CONFIG_SPECS,
    AUDIO_FILENAME_CONFIG_SPECS,
    OUTPUT_FILENAME_CONFIG_SPECS,
    VIDEO_SUFFIXES_CONFIG_SPECS,
    AV_CONFIG_SPECS,
)
CLIP_PROCESSOR_CONFIG_SPECS = merge_config_specs(
    TARGET_ROOT_CONFIG_SPECS,
    ORDER_FILENAME_CONFIG_SPECS,
    OUTPUT_FILENAME_CONFIG_SPECS,
    CLIP_CONFIG_SPECS,
)
VIDEO_INIT_CHECKER_CONFIG_SPECS = merge_config_specs(
    TARGET_ROOT_CONFIG_SPECS,
    ORDER_FILENAME_CONFIG_SPECS,
    VIDEO_SUFFIXES_CONFIG_SPECS,
    OUTPUT_FILENAME_CONFIG_SPECS,
    AV_CONFIG_SPECS,
    PREFIXED_CLIP_CONFIG_SPECS,
)
VIDEO_CLIP_PIPELINE_CONFIG_SPECS = merge_config_specs(
    TARGET_ROOT_CONFIG_SPECS,
    VIDEO_ROOT_CONFIG_SPECS,
    ORDER_FILENAME_CONFIG_SPECS,
    AUDIO_FILENAME_CONFIG_SPECS,
    OUTPUT_FILENAME_CONFIG_SPECS,
    VIDEO_SUFFIXES_CONFIG_SPECS,
    AV_CONFIG_SPECS,
    PREFIXED_CLIP_CONFIG_SPECS,
)


def _resolve_config_path(base_dir: Path, value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def _load_raw_config(
    config_path: Path | None = None,
) -> tuple[Path, dict[str, Any], Path]:
    actual_config_path = config_path or CONFIG_PATH
    base_dir = actual_config_path.parent

    if not actual_config_path.exists():
        raise CheckDataConfigError(f"配置文件不存在: {actual_config_path}")

    try:
        with actual_config_path.open("r", encoding="utf-8") as f:
            raw_config = json.load(f)
    except json.JSONDecodeError as e:
        raise CheckDataConfigError(f"{actual_config_path} JSON 格式错误: {e}") from e

    if not isinstance(raw_config, dict):
        raise CheckDataConfigError(f"{actual_config_path} 根节点必须是对象")

    return actual_config_path, raw_config, base_dir


def _load_optional_value_by_paths(
    raw_config: dict[str, Any],
    *paths: tuple[str, ...],
) -> Any:
    for path in paths:
        current: Any = raw_config
        found = True
        for part in path:
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found:
            return current
    return None


def _optional_nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _optional_filename(value: Any) -> str | None:
    normalized = _optional_nonempty_str(value)
    if normalized is None:
        return None
    if Path(normalized).name != normalized:
        return None
    return normalized


def _optional_resolved_path(base_dir: Path, value: Any) -> str | None:
    normalized = _optional_nonempty_str(value)
    if normalized is None:
        return None
    return _resolve_config_path(base_dir, normalized)


def _optional_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _optional_positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value)
    return None


def _optional_string_tuple(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None

    normalized_items: list[str] = []
    seen_items: set[str] = set()
    for item in value:
        normalized = _optional_nonempty_str(item)
        if normalized is None or normalized in seen_items:
            return None
        seen_items.add(normalized)
        normalized_items.append(normalized)

    if not normalized_items:
        return None
    return tuple(normalized_items)


def _optional_suffix_tuple(value: Any) -> tuple[str, ...] | None:
    items = _optional_string_tuple(value)
    if items is None:
        return None

    normalized_suffixes: list[str] = []
    for item in items:
        if not item.startswith(".") or Path(item).name != item:
            return None
        normalized_suffixes.append(item.lower())
    return tuple(normalized_suffixes)


def _normalize_process_steps(raw_process_steps: Any, source: str) -> tuple[str, ...]:
    if not isinstance(raw_process_steps, list):
        raise CheckDataConfigError(f"{source} 必须是字符串数组")

    normalized_steps: list[str] = []
    seen_steps: set[str] = set()

    for index, item in enumerate(raw_process_steps):
        if not isinstance(item, str):
            raise CheckDataConfigError(f"{source}[{index}] 必须是字符串")

        step = item.strip()
        if not step:
            raise CheckDataConfigError(f"{source}[{index}] 不能为空字符串")

        if step in seen_steps:
            raise CheckDataConfigError(f"{source} 中存在重复步骤: {step}")

        seen_steps.add(step)
        normalized_steps.append(step)

    if not normalized_steps:
        raise CheckDataConfigError(f"{source} 不能为空数组")

    return tuple(normalized_steps)


@dataclass(frozen=True)
class ConfigReader:
    path: Path
    raw: dict[str, Any]
    base_dir: Path

    def get(self, *paths: tuple[str, ...]) -> Any:
        return _load_optional_value_by_paths(self.raw, *paths)

    def _read_value(
        self,
        reader: Callable[[Any], Any],
        *paths: tuple[str, ...],
    ) -> Any:
        return reader(self.get(*paths))

    def _read_path_value(
        self,
        reader: Callable[[Path, Any], Any],
        *paths: tuple[str, ...],
    ) -> Any:
        return reader(self.base_dir, self.get(*paths))

    def read(self, reader_name: str, *paths: tuple[str, ...]) -> Any:
        try:
            reader = getattr(self, reader_name)
        except AttributeError as e:
            raise AttributeError(f"未知配置读取器: {reader_name}") from e
        return reader(*paths)

    def nonempty_str(self, *paths: tuple[str, ...]) -> str | None:
        return self._read_value(_optional_nonempty_str, *paths)

    def filename(self, *paths: tuple[str, ...]) -> str | None:
        return self._read_value(_optional_filename, *paths)

    def resolved_path(self, *paths: tuple[str, ...]) -> str | None:
        return self._read_path_value(_optional_resolved_path, *paths)

    def positive_int(self, *paths: tuple[str, ...]) -> int | None:
        return self._read_value(_optional_positive_int, *paths)

    def nonnegative_int(self, *paths: tuple[str, ...]) -> int | None:
        return self._read_value(_optional_nonnegative_int, *paths)

    def positive_float(self, *paths: tuple[str, ...]) -> float | None:
        return self._read_value(_optional_positive_float, *paths)

    def string_tuple(self, *paths: tuple[str, ...]) -> tuple[str, ...] | None:
        return self._read_value(_optional_string_tuple, *paths)

    def suffix_tuple(self, *paths: tuple[str, ...]) -> tuple[str, ...] | None:
        return self._read_value(_optional_suffix_tuple, *paths)


def load_config(config_path: Path | None = None) -> ConfigReader:
    actual_config_path, raw_config, base_dir = _load_raw_config(config_path)
    return ConfigReader(actual_config_path, raw_config, base_dir)

def load_process_steps_config(config_path: Path | None = None) -> tuple[str, ...] | None:
    config = load_config(config_path)
    raw_process_steps = config.get(
        ("progress", "process_steps"),
        ("check_data", "process_steps"),
    )
    if raw_process_steps is None:
        return None

    return _normalize_process_steps(
        raw_process_steps,
        source=f"{config.path} progress.process_steps",
    )


def load_process_steps_config_or_default(
    config_path: Path | None = None,
    default_steps: Iterable[str] | None = None,
) -> tuple[str, ...]:
    try:
        process_steps = load_process_steps_config(config_path)
        if process_steps is not None:
            return process_steps
    except CheckDataConfigError:
        pass

    if default_steps is None:
        raise CheckDataConfigError("未提供默认 process_steps，且配置中的 process_steps 不可用")

    return tuple(default_steps)


def _filter_builder_kwargs(builder: Callable[..., T], config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        return dict(config)

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(config)

    accepted_keys = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in config.items() if key in accepted_keys}


def build_from_config(
    builder: Callable[..., T],
    loaders: Iterable[Callable[[ConfigReader], Mapping[str, Any]]],
    config_path: Path | None = None,
) -> T:
    config = load_config(config_path)
    merged_config: dict[str, Any] = {}
    for loader in loaders:
        merged_config.update(loader(config))
    return builder(**_filter_builder_kwargs(builder, merged_config))


def build_from_config_or_default(
    builder: Callable[..., T],
    loaders: Iterable[Callable[[ConfigReader], Mapping[str, Any]]],
    config_path: Path | None = None,
    default_builder: Callable[[], T] | None = None,
) -> T:
    try:
        return build_from_config(builder, loaders, config_path)
    except CheckDataConfigError as e:
        fallback_path = config_path or CONFIG_PATH
        print(
            f"\033[31m[error] {fallback_path} 读取失败，改用默认参数: {e} "
            f"config.json参数配置不合法\033[0m"
        )
        return (default_builder or builder)()
