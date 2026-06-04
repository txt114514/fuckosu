from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

from loguru import logger

from Traning.conf import Settings, load_settings


CONFIG_PATH = Path(__file__).resolve().parents[2] / "conf" / "config.yaml"
T = TypeVar("T")


class CheckDataConfigError(Exception):
    pass


class ConfigReader:
    """Small compatibility reader for legacy build_* helpers."""

    def __init__(self, path: Path | None = None):
        self.path = path or CONFIG_PATH
        self.settings = load_settings(self.path)
        self.base_dir = self.path.parent
        self.raw: dict[str, Any] = self.settings.model_dump()

    def get(self, *_paths: tuple[str, ...]) -> Any:
        return None

    def read(self, *_args: Any, **_kwargs: Any) -> Any:
        return None


EMPTY_CONFIG_SPECS: tuple[()] = ()
CHECK_DATA_PIPELINE_CONFIG_SPECS = EMPTY_CONFIG_SPECS
OSU_OSZ_PROCESSOR_CONFIG_SPECS = EMPTY_CONFIG_SPECS
VERIFY_EXPORTER_CONFIG_SPECS = EMPTY_CONFIG_SPECS
VIDEO_PACKAGE_RENAMER_CONFIG_SPECS = EMPTY_CONFIG_SPECS
AV_CORRESPONDENCE_PROCESSOR_CONFIG_SPECS = EMPTY_CONFIG_SPECS
CLIP_PROCESSOR_CONFIG_SPECS = EMPTY_CONFIG_SPECS
VIDEO_INIT_CHECKER_CONFIG_SPECS = EMPTY_CONFIG_SPECS
VIDEO_CLIP_PIPELINE_CONFIG_SPECS = EMPTY_CONFIG_SPECS
AUDIO_MATCH_EXPERIMENT_CONFIG_SPECS = EMPTY_CONFIG_SPECS


def load_config(config_path: Path | None = None) -> ConfigReader:
    return ConfigReader(config_path)


def load_process_steps_config(config_path: Path | None = None) -> tuple[str, ...]:
    return tuple(load_settings(config_path).progress.process_steps)


def load_process_steps_config_or_default(
    config_path: Path | None = None,
    default_steps: Iterable[str] | None = None,
) -> tuple[str, ...]:
    try:
        return load_process_steps_config(config_path)
    except Exception:
        if default_steps is None:
            raise
        return tuple(default_steps)


def _settings_kwargs(settings: Settings) -> dict[str, Any]:
    files = settings.file_management
    return {
        "export_dir": str(files.export_dir),
        "target_root": str(files.target_root),
        "video_root": str(files.video_root),
        "order_filename": files.order_filename,
        "verify_filename": files.verify_filename,
        "difficulty_filename": files.difficulty_filename,
        "verify_failed_filename": files.verify_failed_filename,
        "difficulty_failed_filename": files.difficulty_failed_filename,
        "audio_filename": files.audio_filename,
        "output_filename": files.output_filename,
        "video_suffixes": settings.file_formats.video_suffixes,
        "keyword": settings.file_formats.keyword,
        "status_step": settings.progress.av_status_step,
        "required_steps": settings.progress.av_required_steps,
        "sample_rate": settings.av.sample_rate,
        "envelope_hz": settings.av.envelope_hz,
        "refine_hz": settings.av.refine_hz,
        "refine_search_seconds": settings.av.refine_search_seconds,
        "music_lowpass_hz": settings.av.music_lowpass_hz,
        "global_offset_ms": settings.global_offset_ms,
        "use_audio_match_experiment": settings.video_clip.use_audio_match_experiment,
        "clip_failed_filename": files.clip_failed_filename,
        "clip_status_step": settings.clip.status_step,
        "clip_required_steps": settings.clip.required_steps,
        "clip_crop_reference_width": settings.clip.crop_reference_width,
        "clip_crop_reference_height": settings.clip.crop_reference_height,
        "clip_crop_left": settings.clip.crop_left,
        "clip_crop_top": settings.clip.crop_top,
        "clip_crop_right": settings.clip.crop_right,
        "clip_crop_bottom": settings.clip.crop_bottom,
        "crop_reference_width": settings.clip.crop_reference_width,
        "crop_reference_height": settings.clip.crop_reference_height,
        "crop_left": settings.clip.crop_left,
        "crop_top": settings.clip.crop_top,
        "crop_right": settings.clip.crop_right,
        "crop_bottom": settings.clip.crop_bottom,
    }


def _filter_builder_kwargs(builder: Callable[..., T], config: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(builder)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in config.items() if key in accepted}


def build_from_config(
    builder: Callable[..., T],
    _loaders: Iterable[Callable[[ConfigReader], Mapping[str, Any]]],
    config_path: Path | None = None,
) -> T:
    settings = load_settings(config_path)
    return builder(**_filter_builder_kwargs(builder, _settings_kwargs(settings)))


def build_from_config_or_default(
    builder: Callable[..., T],
    loaders: Iterable[Callable[[ConfigReader], Mapping[str, Any]]],
    config_path: Path | None = None,
    default_builder: Callable[[], T] | None = None,
) -> T:
    try:
        return build_from_config(builder, loaders, config_path)
    except Exception as e:
        logger.error("{} 读取失败，改用默认参数: {}", config_path or CONFIG_PATH, e)
        return (default_builder or builder)()
