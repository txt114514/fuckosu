from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
DataSplit = Literal["all", "train", "validation"]


class SettingsError(Exception):
    pass


class RuntimeSettings(BaseModel):
    seed: int = 2026
    device: str = "cuda"


class InputSettings(BaseModel):
    width: int = 1484
    height: int = 846
    resize: bool = False

    @field_validator("width", "height")
    @classmethod
    def _positive_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("input dimensions must be positive")
        return value


class TilingConfig(BaseModel):
    patch_height: int = 512
    patch_width: int = 512
    overlap_y: int = 128
    overlap_x: int = 128
    patch_batch_size: int = 1
    serial: bool = True

    @field_validator("patch_height", "patch_width", "patch_batch_size")
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("tiling dimensions and batch size must be positive")
        return value

    @field_validator("overlap_y", "overlap_x")
    @classmethod
    def _nonnegative_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("tiling overlap must be nonnegative")
        return value

    @model_validator(mode="after")
    def validate_tiling(self) -> TilingConfig:
        if self.overlap_x >= self.patch_width:
            raise ValueError("overlap_x must be smaller than patch_width")
        if self.overlap_y >= self.patch_height:
            raise ValueError("overlap_y must be smaller than patch_height")
        if self.serial and self.patch_batch_size != 1:
            raise ValueError("serial patch processing requires patch_batch_size=1")
        return self


class LocalEncoderConfig(BaseModel):
    stem_channels: int = 8
    feature_channels: int = 48
    output_stride: int = 8
    embedding_dim: int = 96

    @field_validator(
        "stem_channels",
        "feature_channels",
        "output_stride",
        "embedding_dim",
    )
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("local encoder dimensions must be positive")
        return value


class GlobalEncoderConfig(BaseModel):
    input_height: int = 360
    input_width: int = 640
    feature_channels: int = 64
    backbone: Literal[
        "lightweight_cnn",
        "mobilenet_v3_small",
        "convnext_atto",
        "dinov3_external",
    ] = "lightweight_cnn"
    pretrained: bool = False
    frozen: bool = False

    @field_validator("input_height", "input_width", "feature_channels")
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("global encoder dimensions must be positive")
        return value


class FusionConfig(BaseModel):
    mode: Literal["disabled", "gated", "gated_sparse_sampling"] = (
        "gated_sparse_sampling"
    )
    heads: int = 4
    sampling_points: int = 4
    layers: int = 2
    hidden_dim: int = 96

    @field_validator("heads", "sampling_points", "layers", "hidden_dim")
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("fusion dimensions must be positive")
        return value

    @model_validator(mode="after")
    def validate_attention_shape(self) -> FusionConfig:
        if self.hidden_dim % self.heads != 0:
            raise ValueError("fusion hidden_dim must be divisible by heads")
        return self


class TemporalConfig(BaseModel):
    enabled: bool = True
    model_type: Literal["causal_gru"] = "causal_gru"
    hidden_size: int = 256
    layers: int = 2
    history_frames: int = 8

    @field_validator("hidden_size", "layers", "history_frames")
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("temporal dimensions must be positive")
        return value


class MemoryConfig(BaseModel):
    amp_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    gradient_checkpointing: bool = True
    backward_per_patch: bool = True
    cache_global_features: bool = True
    offload_candidates_to_cpu: bool = True
    max_vram_gib: float = 7.5

    @field_validator("max_vram_gib")
    @classmethod
    def _positive_vram(cls, value: float) -> float:
        if value <= 0 or value != value or value == float("inf"):
            raise ValueError("max_vram_gib must be finite and positive")
        return value


class SMETConfig(BaseModel):
    enabled: bool = False


class LoaderSettings(BaseModel):
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = True

    @field_validator("batch_size")
    @classmethod
    def _positive_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size must be positive")
        return value

    @field_validator("num_workers")
    @classmethod
    def _nonnegative_workers(cls, value: int) -> int:
        if value < 0:
            raise ValueError("num_workers must be nonnegative")
        return value


class EvaluationSettings(BaseModel):
    min_click_interval_ms: float = 50.0

    @field_validator("min_click_interval_ms")
    @classmethod
    def _nonnegative_interval(cls, value: float) -> float:
        if value < 0 or value != value or value == float("inf"):
            raise ValueError("min_click_interval_ms must be finite and nonnegative")
        return value


class VisualizationSettings(BaseModel):
    enabled: bool = False
    every_n_steps: int = 500
    output_dir: Path = REPO_ROOT / "traning_example"
    save_images: bool = True
    gallery_samples_per_group: int = 10
    show_window: bool = False
    window_title: str = "osu! training annotation"
    ffplay_binary: str = "ffplay"

    @field_validator("every_n_steps", "gallery_samples_per_group")
    @classmethod
    def _positive_interval(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("visualization intervals and counts must be positive")
        return value


class DataInputSettings(BaseModel):
    dataset_root: Path = REPO_ROOT / "training_package" / "video_segments"
    dimensions: tuple[str, ...] = ("atomic", "long_sequence")
    categories: tuple[str, ...] = ()
    include_items: tuple[str, ...] = ()
    exclude_items: tuple[str, ...] = ()
    train_items: tuple[str, ...] = ()
    validation_items: tuple[str, ...] = ()
    sample_fps: float = 60.0
    frame_step: int = 1
    max_segments: int | None = None
    max_frames_per_segment: int | None = None
    visibility_post_ms: float = 100.0
    normalize_images: bool = True
    strict: bool = True
    patch_width: int = 512
    patch_height: int = 512
    overlap_x: int = 128
    overlap_y: int = 128

    @field_validator("sample_fps")
    @classmethod
    def _positive_fps(cls, value: float) -> float:
        if value <= 0 or value != value or value == float("inf"):
            raise ValueError("sample_fps must be finite and positive")
        return value

    @field_validator("frame_step", "patch_width", "patch_height")
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("max_segments", "max_frames_per_segment")
    @classmethod
    def _optional_positive_integer(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("optional limits must be positive")
        return value

    @field_validator("visibility_post_ms")
    @classmethod
    def _nonnegative_visibility(cls, value: float) -> float:
        if value < 0 or value != value or value == float("inf"):
            raise ValueError("visibility_post_ms must be finite and nonnegative")
        return value

    @field_validator(
        "dimensions",
        "categories",
        "include_items",
        "exclude_items",
        "train_items",
        "validation_items",
    )
    @classmethod
    def _unique_nonempty_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item for item in cleaned):
            raise ValueError("filter values must be nonempty strings")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("filter values must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_item_splits(self) -> DataInputSettings:
        overlap = set(self.train_items) & set(self.validation_items)
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(f"train_items and validation_items overlap: {joined}")
        return self

    def validate_tiling(self) -> None:
        if not 0 <= self.overlap_x < self.patch_width:
            raise ValueError("overlap_x must be in [0, patch_width)")
        if not 0 <= self.overlap_y < self.patch_height:
            raise ValueError("overlap_y must be in [0, patch_height)")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OSU_TRAINING_",
        env_nested_delimiter="__",
        extra="ignore",
        validate_assignment=True,
    )

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    input: InputSettings = Field(default_factory=InputSettings)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    local_encoder: LocalEncoderConfig = Field(default_factory=LocalEncoderConfig)
    global_encoder: GlobalEncoderConfig = Field(default_factory=GlobalEncoderConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    smet: SMETConfig = Field(default_factory=SMETConfig)
    data_input: DataInputSettings = Field(default_factory=DataInputSettings)
    loader: LoaderSettings = Field(default_factory=LoaderSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )


def _read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise SettingsError(f"config file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise SettingsError(f"failed to read config: {config_path}") from error
    if not isinstance(raw, dict):
        raise SettingsError("config root must be a mapping")
    return raw


def _resolve_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = dict(raw)
    data_input = dict(resolved.get("data_input") or {})
    dataset_root = data_input.get("dataset_root")
    if dataset_root is not None:
        path = Path(dataset_root).expanduser()
        data_input["dataset_root"] = (
            path if path.is_absolute() else (base_dir / path).resolve()
        )
    resolved["data_input"] = data_input
    visualization = dict(resolved.get("visualization") or {})
    output_dir = visualization.get("output_dir")
    if output_dir is not None:
        path = Path(output_dir).expanduser()
        visualization["output_dir"] = (
            path if path.is_absolute() else (base_dir / path).resolve()
        )
    resolved["visualization"] = visualization
    return resolved


def load_settings(config_path: Path | None = None) -> Settings:
    selected = (config_path or CONFIG_PATH).resolve()
    raw = _resolve_paths(_read_config(selected), selected.parent)
    try:
        settings = Settings(**raw)
        settings.data_input.validate_tiling()
        settings.tiling.validate_tiling()
        return settings
    except (ValidationError, ValueError) as error:
        raise SettingsError(f"invalid training config: {selected}") from error


__all__ = [
    "CONFIG_PATH",
    "DataSplit",
    "DataInputSettings",
    "EvaluationSettings",
    "FusionConfig",
    "GlobalEncoderConfig",
    "InputSettings",
    "LoaderSettings",
    "LocalEncoderConfig",
    "MemoryConfig",
    "REPO_ROOT",
    "RuntimeSettings",
    "SMETConfig",
    "Settings",
    "SettingsError",
    "TemporalConfig",
    "TilingConfig",
    "VisualizationSettings",
    "load_settings",
]
