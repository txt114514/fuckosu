from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SETTING_PATH = Path(__file__).resolve().parent / "setting.json"
DEFAULT_USE_AUDIO_MATCH_EXPERIMENT = True
DEFAULT_GLOBAL_OFFSET_MS = 0.0


class TrainingSettingError(Exception):
    pass


@dataclass(frozen=True)
class TrainingRunSettings:
    overwrite: bool = False
    continue_on_error: bool = False
    run_check_data: bool = True
    run_get_files: bool = True
    run_verify_export: bool = True
    run_difficulty_export: bool = True
    run_video_clip: bool = True
    run_video_init_check: bool = True
    run_video_match: bool = True
    run_av_correspondence: bool = True
    run_clip_stage: bool = True
    use_audio_match_experiment: bool = DEFAULT_USE_AUDIO_MATCH_EXPERIMENT
    global_offset_ms: float = DEFAULT_GLOBAL_OFFSET_MS


def _read_setting_section(
    raw_settings: dict[str, Any],
    key: str,
    source: Path,
) -> dict[str, Any]:
    value = raw_settings.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TrainingSettingError(f"{source} 中的 {key} 必须是对象")
    return value


def _read_setting_bool(
    section: dict[str, Any],
    key: str,
    default: bool,
    source: str,
) -> bool:
    value = section.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TrainingSettingError(f"{source}.{key} 必须是 true 或 false")
    return value


def _read_setting_float(
    section: dict[str, Any],
    key: str,
    default: float,
    source: str,
) -> float:
    value = section.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingSettingError(f"{source}.{key} 必须是数字")

    normalized = float(value)
    if not math.isfinite(normalized):
        raise TrainingSettingError(f"{source}.{key} 必须是有限数字")
    return normalized


def load_training_settings(setting_path: Path | None = None) -> TrainingRunSettings:
    actual_setting_path = setting_path or SETTING_PATH
    defaults = TrainingRunSettings()

    if not actual_setting_path.exists():
        return defaults

    try:
        with actual_setting_path.open("r", encoding="utf-8") as f:
            raw_settings = json.load(f)
    except json.JSONDecodeError as e:
        raise TrainingSettingError(f"{actual_setting_path} JSON 格式错误: {e}") from e

    if not isinstance(raw_settings, dict):
        raise TrainingSettingError(f"{actual_setting_path} 根节点必须是对象")

    runtime_settings = _read_setting_section(raw_settings, "runtime", actual_setting_path)
    check_data_settings = _read_setting_section(raw_settings, "check_data", actual_setting_path)
    video_clip_settings = _read_setting_section(raw_settings, "video_clip", actual_setting_path)

    return TrainingRunSettings(
        overwrite=_read_setting_bool(
            runtime_settings,
            "overwrite",
            defaults.overwrite,
            "runtime",
        ),
        continue_on_error=_read_setting_bool(
            runtime_settings,
            "continue_on_error",
            defaults.continue_on_error,
            "runtime",
        ),
        run_check_data=_read_setting_bool(
            check_data_settings,
            "enabled",
            defaults.run_check_data,
            "check_data",
        ),
        run_get_files=_read_setting_bool(
            check_data_settings,
            "run_get_files",
            defaults.run_get_files,
            "check_data",
        ),
        run_verify_export=_read_setting_bool(
            check_data_settings,
            "run_verify_export",
            defaults.run_verify_export,
            "check_data",
        ),
        run_difficulty_export=_read_setting_bool(
            check_data_settings,
            "run_difficulty_export",
            defaults.run_difficulty_export,
            "check_data",
        ),
        run_video_clip=_read_setting_bool(
            video_clip_settings,
            "enabled",
            defaults.run_video_clip,
            "video_clip",
        ),
        run_video_init_check=_read_setting_bool(
            video_clip_settings,
            "run_video_init_check",
            defaults.run_video_init_check,
            "video_clip",
        ),
        run_video_match=_read_setting_bool(
            video_clip_settings,
            "run_video_match",
            defaults.run_video_match,
            "video_clip",
        ),
        run_av_correspondence=_read_setting_bool(
            video_clip_settings,
            "run_av_correspondence",
            defaults.run_av_correspondence,
            "video_clip",
        ),
        run_clip_stage=_read_setting_bool(
            video_clip_settings,
            "run_clip_stage",
            defaults.run_clip_stage,
            "video_clip",
        ),
        use_audio_match_experiment=_read_setting_bool(
            video_clip_settings,
            "use_audio_match_experiment",
            defaults.use_audio_match_experiment,
            "video_clip",
        ),
        global_offset_ms=_read_setting_float(
            video_clip_settings,
            "global_offset_ms",
            defaults.global_offset_ms,
            "video_clip",
        ),
    )


def load_training_settings_or_default(
    setting_path: Path | None = None,
) -> TrainingRunSettings:
    actual_setting_path = setting_path or SETTING_PATH
    try:
        return load_training_settings(setting_path=setting_path)
    except TrainingSettingError as e:
        print(f"\033[31m[error] {actual_setting_path} 读取失败，改用默认运行设置: {e}\033[0m")
        return TrainingRunSettings()
