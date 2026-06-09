from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class SettingsError(Exception):
    pass


class RuntimeSettings(BaseModel):
    overwrite: bool = False
    continue_on_error: bool = False


class CheckDataSettings(BaseModel):
    run_get_files: bool = True
    run_verify_export: bool = True
    run_difficulty_export: bool = True


class VideoClipSettings(BaseModel):
    run_video_match: bool = True
    run_av_correspondence: bool = True
    run_clip_stage: bool = True
    use_audio_match_experiment: bool = True
    global_offset_ms: float = 0.0

    @field_validator("global_offset_ms")
    @classmethod
    def _finite_offset(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("global_offset_ms must be finite")
        return value


class FileManagementSettings(BaseModel):
    export_dir: Path = REPO_ROOT / "osu-lazer" / "exports"
    target_root: Path = REPO_ROOT / "training_package" / "match-completed_package"
    video_root: Path = REPO_ROOT / "training_package" / "video_package"
    order_filename: str = "order.txt"
    verify_filename: str = "verify.txt"
    difficulty_filename: str = "difficulty.txt"
    verify_failed_filename: str = "verify_failed.txt"
    difficulty_failed_filename: str = "difficulty_failed.txt"
    audio_filename: str = "audio.mp3"
    output_filename: str = "video_processed.mp4"
    av_correspondence_failed_filename: str = "av_correspondence_failed.txt"
    clip_failed_filename: str = "clip_failed.txt"


class FileFormatSettings(BaseModel):
    keyword: str = "normal"
    video_suffixes: tuple[str, ...] = (".mp4", ".webm", ".mkv", ".avi", ".mov")


class AVSettings(BaseModel):
    sample_rate: int = 8000
    envelope_hz: int = 100
    refine_hz: int = 1000
    refine_search_seconds: float = 1.5
    music_lowpass_hz: int = 1500
    verify_correction_window_ms: float = 120.0


class AudioMatchSettings(BaseModel):
    top_k: int = 3
    match_status_step: str = "video_matched"


class PackageSettings(BaseModel):
    ignore_patterns: tuple[str, ...] = (
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "temp/",
        "tmp/",
    )


class ClipSettings(BaseModel):
    crop_reference_width: int = 2048
    crop_reference_height: int = 1152
    crop_left: int = 186
    crop_top: int = 178
    crop_right: int = 1768
    crop_bottom: int = 1080
    status_step: str = "video_processed"
    required_steps: tuple[str, ...] = ("av_corresponded",)


class ProgressSettings(BaseModel):
    process_steps: tuple[str, ...] = (
        "osu_imported",
        "audio_imported",
        "verify_exported",
        "difficulty_exported",
        "video_matched",
        "av_corresponded",
        "video_processed",
    )
    av_status_step: str = "av_corresponded"
    av_required_steps: tuple[str, ...] = ("audio_imported", "video_matched")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRAINING_",
        extra="ignore",
        validate_assignment=True,
    )

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    check_data: CheckDataSettings = Field(default_factory=CheckDataSettings)
    video_clip: VideoClipSettings = Field(default_factory=VideoClipSettings)
    file_management: FileManagementSettings = Field(default_factory=FileManagementSettings)
    file_formats: FileFormatSettings = Field(default_factory=FileFormatSettings)
    av: AVSettings = Field(default_factory=AVSettings)
    audio_match: AudioMatchSettings = Field(default_factory=AudioMatchSettings)
    package: PackageSettings = Field(default_factory=PackageSettings)
    clip: ClipSettings = Field(default_factory=ClipSettings)
    progress: ProgressSettings = Field(default_factory=ProgressSettings)

    @property
    def target_root(self) -> Path:
        return self.file_management.target_root

    @property
    def overwrite(self) -> bool:
        return self.runtime.overwrite

    @property
    def continue_on_error(self) -> bool:
        return self.runtime.continue_on_error

    @property
    def global_offset_ms(self) -> float:
        return self.video_clip.global_offset_ms


def _resolve_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    file_management = raw.get("file_management")
    if not isinstance(file_management, dict):
        return raw

    for key in ("export_dir", "target_root", "video_root"):
        value = file_management.get(key)
        if isinstance(value, str):
            path = Path(value)
            file_management[key] = path if path.is_absolute() else (base_dir / path).resolve()
    return raw


def _extract_nested(raw: dict[str, Any]) -> dict[str, Any]:
    parameters = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
    progress = raw.get("progress") if isinstance(raw.get("progress"), dict) else {}
    status_steps = (
        progress.get("status_steps")
        if isinstance(progress.get("status_steps"), dict)
        else {}
    )
    required_steps = (
        progress.get("required_steps")
        if isinstance(progress.get("required_steps"), dict)
        else {}
    )

    extracted = dict(raw)
    extracted["av"] = parameters.get("av_correspondence", {})
    extracted["audio_match"] = parameters.get("audio_match", {})
    extracted["clip"] = {
        **parameters.get("clip", {}),
        "status_step": status_steps.get("clip", "video_processed"),
        "required_steps": required_steps.get("clip", ("av_corresponded",)),
    }
    extracted["progress"] = {
        "process_steps": progress.get("process_steps", ProgressSettings().process_steps),
        "av_status_step": status_steps.get("av_correspondence", "av_corresponded"),
        "av_required_steps": required_steps.get(
            "av_correspondence",
            ("audio_imported", "video_matched"),
        ),
    }
    return extracted


def _read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            if config_path.suffix.lower() in {".yaml", ".yml"}:
                payload = yaml.safe_load(f)
            else:
                payload = json.load(f)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise SettingsError(f"{config_path} 配置格式错误: {e}") from e

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SettingsError(f"{config_path} 根节点必须是对象")
    return payload


def load_settings(config_path: Path | None = None) -> Settings:
    actual_path = config_path or CONFIG_PATH
    raw = _extract_nested(_resolve_paths(_read_config(actual_path), actual_path.parent))
    try:
        return Settings(**raw)
    except ValidationError as e:
        raise SettingsError(str(e)) from e
