from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
T = TypeVar("T")


class CheckDataConfigError(Exception):
    pass


def _strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments while preserving string contents."""
    result: list[str] = []
    in_string = False
    in_single_line_comment = False
    in_multi_line_comment = False
    escape = False
    index = 0
    length = len(text)

    while index < length:
        ch = text[index]
        next_ch = text[index + 1] if index + 1 < length else ""

        if in_single_line_comment:
            if ch == "\n":
                in_single_line_comment = False
                result.append(ch)
            index += 1
            continue

        if in_multi_line_comment:
            if ch == "*" and next_ch == "/":
                in_multi_line_comment = False
                index += 2
                continue
            if ch == "\n":
                result.append(ch)
            index += 1
            continue

        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            index += 1
            continue

        if ch == '"':
            in_string = True
            result.append(ch)
            index += 1
            continue

        if ch == "/" and next_ch == "/":
            in_single_line_comment = True
            index += 2
            continue

        if ch == "/" and next_ch == "*":
            in_multi_line_comment = True
            index += 2
            continue

        result.append(ch)
        index += 1

    return "".join(result)


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
            raw_text = f.read()
        raw_config = json.loads(_strip_json_comments(raw_text))
    except json.JSONDecodeError as e:
        raise CheckDataConfigError(
            f"{actual_config_path} JSON 格式错误(支持 // 与 /* */ 注释): {e}"
        ) from e

    if not isinstance(raw_config, dict):
        raise CheckDataConfigError(f"{actual_config_path} 根节点必须是对象")

    return actual_config_path, raw_config, base_dir


def _load_optional_section(
    raw_config: dict[str, Any],
    section_name: str,
) -> dict[str, Any] | None:
    section = raw_config.get(section_name)
    if section is None:
        return None
    if not isinstance(section, dict):
        return None
    return section


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


def load_check_data_config(config_path: Path | None = None) -> dict[str, Any]:
    _actual_config_path, raw_config, base_dir = _load_raw_config(config_path)
    section = _load_optional_section(raw_config, "check_data")
    if section is None:
        return {}

    config: dict[str, Any] = {}
    export_dir = _optional_resolved_path(base_dir, section.get("export_dir"))
    target_root = _optional_resolved_path(base_dir, section.get("target_root"))
    keyword = _optional_nonempty_str(section.get("keyword"))

    if export_dir is not None:
        config["export_dir"] = export_dir
    if target_root is not None:
        config["target_root"] = target_root
    if keyword is not None:
        config["keyword"] = keyword

    return config


def load_process_steps_config(config_path: Path | None = None) -> tuple[str, ...] | None:
    actual_config_path, raw_config, _base_dir = _load_raw_config(config_path)
    section = _load_optional_section(raw_config, "check_data")
    if section is None:
        return None

    raw_process_steps = section.get("process_steps")
    if raw_process_steps is None:
        return None

    return _normalize_process_steps(
        raw_process_steps,
        source=f"{actual_config_path} check_data.process_steps",
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


def load_get_check_data_files_config(config_path: Path | None = None) -> dict[str, Any]:
    _actual_config_path, raw_config, _base_dir = _load_raw_config(config_path)
    section = _load_optional_section(raw_config, "get_check_data")
    if section is None:
        return {}

    config: dict[str, Any] = {}
    order_filename = _optional_filename(section.get("order_filename"))
    verify_filename = _optional_filename(section.get("verify_filename"))
    difficulty_filename = _optional_filename(section.get("difficulty_filename"))
    verify_failed_filename = _optional_filename(section.get("verify_failed_filename"))
    difficulty_failed_filename = _optional_filename(
        section.get("difficulty_failed_filename")
    )

    if order_filename is not None:
        config["order_filename"] = order_filename
    if verify_filename is not None:
        config["verify_filename"] = verify_filename
    if difficulty_filename is not None:
        config["difficulty_filename"] = difficulty_filename
    if verify_failed_filename is not None:
        config["verify_failed_filename"] = verify_failed_filename
    if difficulty_failed_filename is not None:
        config["difficulty_failed_filename"] = difficulty_failed_filename

    return config


def load_video_shared_config(config_path: Path | None = None) -> dict[str, Any]:
    _actual_config_path, raw_config, _base_dir = _load_raw_config(config_path)
    section = _load_optional_section(raw_config, "video_shared")
    if section is None:
        return {}

    config: dict[str, Any] = {}
    video_suffixes = _optional_suffix_tuple(section.get("video_suffixes"))
    audio_filename = _optional_filename(section.get("audio_filename"))
    output_filename = _optional_filename(section.get("output_filename"))
    status_step = _optional_nonempty_str(section.get("status_step"))

    if video_suffixes is not None:
        config["video_suffixes"] = video_suffixes
    if audio_filename is not None:
        config["audio_filename"] = audio_filename
    if output_filename is not None:
        config["output_filename"] = output_filename
    if status_step is not None:
        config["status_step"] = status_step

    return config


def load_video_match_config(config_path: Path | None = None) -> dict[str, Any]:
    _actual_config_path, raw_config, base_dir = _load_raw_config(config_path)
    section = _load_optional_section(raw_config, "video_match")
    if section is None:
        return {}

    config: dict[str, Any] = {}
    video_root = _optional_resolved_path(base_dir, section.get("video_root"))
    if video_root is not None:
        config["video_root"] = video_root
    return config


def load_av_correspondence_config(config_path: Path | None = None) -> dict[str, Any]:
    _actual_config_path, raw_config, _base_dir = _load_raw_config(config_path)
    section = _load_optional_section(raw_config, "av_correspondence")
    if section is None:
        return {}

    config: dict[str, Any] = {}
    failed_filename = _optional_filename(section.get("failed_filename"))
    required_steps = _optional_string_tuple(section.get("required_steps"))
    sample_rate = _optional_positive_int(section.get("sample_rate"))
    envelope_hz = _optional_positive_int(section.get("envelope_hz"))
    refine_hz = _optional_positive_int(section.get("refine_hz"))
    refine_search_seconds = _optional_positive_float(section.get("refine_search_seconds"))

    if failed_filename is not None:
        config["failed_filename"] = failed_filename
    if required_steps is not None:
        config["required_steps"] = required_steps
    if sample_rate is not None:
        config["sample_rate"] = sample_rate
    if envelope_hz is not None:
        config["envelope_hz"] = envelope_hz
    if refine_hz is not None:
        config["refine_hz"] = refine_hz
    if refine_search_seconds is not None:
        config["refine_search_seconds"] = refine_search_seconds

    return config


def _filter_builder_kwargs(builder: Callable[..., T], config: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        return config

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return config

    accepted_keys = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in config.items() if key in accepted_keys}


def _build_from_config_loaders(
    builder: Callable[..., T],
    loaders: Iterable[Callable[[Path | None], dict[str, Any]]],
    config_path: Path | None = None,
) -> T:
    merged_config: dict[str, Any] = {}
    for loader in loaders:
        merged_config.update(loader(config_path))
    return builder(**_filter_builder_kwargs(builder, merged_config))


def _build_from_config_loaders_or_default(
    builder: Callable[..., T],
    loaders: Iterable[Callable[[Path | None], dict[str, Any]]],
    config_path: Path | None = None,
    default_builder: Callable[[], T] | None = None,
) -> T:
    try:
        return _build_from_config_loaders(builder, loaders, config_path)
    except CheckDataConfigError as e:
        fallback_path = config_path or CONFIG_PATH
        print(
            f"\033[31m[error] {fallback_path} 读取失败，改用默认参数: {e} "
            f"config.json参数配置不合法\033[0m"
        )
        return (default_builder or builder)()


def build_from_check_data_config(
    builder: Callable[..., T],
    config_path: Path | None = None,
) -> T:
    return _build_from_config_loaders(builder, [load_check_data_config], config_path)


def build_from_check_data_config_or_default(
    builder: Callable[..., T],
    config_path: Path | None = None,
    default_builder: Callable[[], T] | None = None,
) -> T:
    return _build_from_config_loaders_or_default(
        builder,
        [load_check_data_config],
        config_path,
        default_builder,
    )


def build_from_get_check_data_config_or_default(
    builder: Callable[..., T],
    config_path: Path | None = None,
    default_builder: Callable[[], T] | None = None,
) -> T:
    return _build_from_config_loaders_or_default(
        builder,
        [
            load_check_data_config,
            load_video_shared_config,
            load_get_check_data_files_config,
        ],
        config_path,
        default_builder,
    )


def build_from_video_shared_config_or_default(
    builder: Callable[..., T],
    config_path: Path | None = None,
    default_builder: Callable[[], T] | None = None,
) -> T:
    return _build_from_config_loaders_or_default(
        builder,
        [load_check_data_config, load_video_shared_config],
        config_path,
        default_builder,
    )


def build_from_video_match_config_or_default(
    builder: Callable[..., T],
    config_path: Path | None = None,
    default_builder: Callable[[], T] | None = None,
) -> T:
    return _build_from_config_loaders_or_default(
        builder,
        [load_check_data_config, load_video_shared_config, load_video_match_config],
        config_path,
        default_builder,
    )


def build_from_av_correspondence_config_or_default(
    builder: Callable[..., T],
    config_path: Path | None = None,
    default_builder: Callable[[], T] | None = None,
) -> T:
    return _build_from_config_loaders_or_default(
        builder,
        [
            load_check_data_config,
            load_video_shared_config,
            load_av_correspondence_config,
        ],
        config_path,
        default_builder,
    )


def build_from_video_clip_config_or_default(
    builder: Callable[..., T],
    config_path: Path | None = None,
    default_builder: Callable[[], T] | None = None,
) -> T:
    return _build_from_config_loaders_or_default(
        builder,
        [
            load_check_data_config,
            load_video_shared_config,
            load_video_match_config,
            load_av_correspondence_config,
        ],
        config_path,
        default_builder,
    )
