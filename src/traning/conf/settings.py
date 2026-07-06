from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from package.coordinates import COORDINATE_TRANSFORM_VERSION
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
DataSplit = Literal["all", "train", "validation", "test"]


class SettingsError(Exception):
    pass


class RuntimeSettings(BaseModel):
    seed: int = 2026
    device: str = "cuda"


class InputSettings(BaseModel):
    width: int = 1484
    height: int = 846
    resize: bool = False
    color_cues: Literal["disabled", "osu_basic"] = "disabled"

    @field_validator("width", "height")
    @classmethod
    def _positive_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("input dimensions must be positive")
        return value


class PlayfieldRectSettings(BaseModel):
    left: float
    top: float
    width: float
    height: float

    @field_validator("left", "top", "width", "height")
    @classmethod
    def _finite_number(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("playfield rect values must be finite")
        return value

    @field_validator("width", "height")
    @classmethod
    def _positive_dimension(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("playfield rect dimensions must be positive")
        return value


class CoordinateTransformSettings(BaseModel):
    version: str = COORDINATE_TRANSFORM_VERSION
    mode: Literal["explicit_rect", "legacy_centered"] = "legacy_centered"
    playfield_rect: PlayfieldRectSettings | None = None

    @model_validator(mode="after")
    def validate_transform(self) -> CoordinateTransformSettings:
        if self.version != COORDINATE_TRANSFORM_VERSION:
            raise ValueError(
                f"coordinate transform version must be {COORDINATE_TRANSFORM_VERSION}"
            )
        if self.mode == "explicit_rect" and self.playfield_rect is None:
            raise ValueError("explicit_rect mode requires playfield_rect")
        return self


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


class TemporalLossWeights(BaseModel):
    action: float = 1.0
    candidate: float = 1.0
    xy: float = 1.0
    time_offset: float = 0.01

    @field_validator("action", "candidate", "xy", "time_offset")
    @classmethod
    def _finite_nonnegative(cls, value: float) -> float:
        if value < 0 or value != value or value == float("inf"):
            raise ValueError("temporal loss weights must be finite and nonnegative")
        return value


class SpatialConsistencyLossWeights(BaseModel):
    embedding: float = 0.0
    ring_radius: float = 0.0
    slider_continuity: float = 0.0

    @field_validator("embedding", "ring_radius", "slider_continuity")
    @classmethod
    def _finite_nonnegative(cls, value: float) -> float:
        if value < 0 or value != value or value == float("inf"):
            raise ValueError("spatial consistency loss weights must be finite and nonnegative")
        return value


class TrainingSettings(BaseModel):
    temporal_loss_weights: TemporalLossWeights = Field(
        default_factory=TemporalLossWeights
    )
    spatial_consistency_loss_weights: SpatialConsistencyLossWeights = Field(
        default_factory=SpatialConsistencyLossWeights
    )


class MemoryConfig(BaseModel):
    amp_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    gradient_checkpointing: bool = True
    backward_per_patch: bool = True
    cache_global_features: bool = True
    offload_candidates_to_cpu: bool = True
    channels_last: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    matmul_float32_precision: Literal["highest", "high", "medium"] = "high"
    grad_scaler: Literal["auto", "enabled", "disabled"] = "auto"
    compile_model: bool = False
    max_vram_gib: float = 6.5
    reserve_vram_gib: float = 1.0
    max_ram_gib: float | None = 24.0
    reserve_ram_gib: float = 4.0

    @field_validator("reserve_vram_gib", "reserve_ram_gib")
    @classmethod
    def _finite_nonnegative_memory(cls, value: float) -> float:
        if value < 0 or value != value or value == float("inf"):
            raise ValueError("memory values must be finite and nonnegative")
        return value

    @field_validator("max_vram_gib")
    @classmethod
    def _positive_memory(cls, value: float) -> float:
        if value <= 0 or value != value or value == float("inf"):
            raise ValueError("max_vram_gib must be finite and positive")
        return value

    @field_validator("max_ram_gib")
    @classmethod
    def _optional_positive_memory(cls, value: float | None) -> float | None:
        if value is not None and (value <= 0 or value != value or value == float("inf")):
            raise ValueError("max_ram_gib must be finite and positive when set")
        return value


class SMETConfig(BaseModel):
    enabled: bool = False
    sparsity: float = 0.50
    update_interval: int = 16
    min_density: float = 0.05

    @field_validator("sparsity")
    @classmethod
    def _sparsity(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("SMET sparsity must be in [0, 1)")
        return value

    @field_validator("update_interval")
    @classmethod
    def _positive_interval(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("SMET update_interval must be positive")
        return value

    @field_validator("min_density")
    @classmethod
    def _density(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("SMET min_density must be in (0, 1]")
        return value


class CandidateCacheSettings(BaseModel):
    output_root: Path = REPO_ROOT / "runs" / "candidate_cache"
    max_candidates_per_frame: int = 32
    save_dtype: Literal["float32", "float16"] = "float16"
    storage: Literal["jsonl"] = "jsonl"
    score_threshold: float = 0.05
    nms_radius_px: float = 32.0
    slider_threshold: float = 0.5
    slider_min_cells: int = 4
    slider_path_points: int = 32
    max_slider_paths: int = 16
    low_confidence_threshold: float = 0.60
    close_score_margin: float = 0.05
    slider_attach_distance_px: float = 48.0
    local_refiner_enabled: bool = False
    local_refiner_top_k: int = 4
    local_refiner_radius_px: float = 12.0
    ambiguity_review_enabled: bool = True
    ambiguity_review_max_candidates: int = 8

    @field_validator(
        "max_candidates_per_frame",
        "slider_min_cells",
        "slider_path_points",
        "max_slider_paths",
        "local_refiner_top_k",
    )
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("candidate cache integer settings must be positive")
        return value

    @field_validator("score_threshold", "slider_threshold")
    @classmethod
    def _probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("candidate cache thresholds must be in [0, 1]")
        return value

    @field_validator(
        "nms_radius_px",
        "low_confidence_threshold",
        "close_score_margin",
        "slider_attach_distance_px",
        "local_refiner_radius_px",
    )
    @classmethod
    def _nonnegative_float(cls, value: float) -> float:
        if value < 0 or value != value or value == float("inf"):
            raise ValueError("candidate cache float settings must be finite")
        return value


class OptimizationSettings(BaseModel):
    enabled: bool = False
    max_generated_jobs: int = 1
    execute_generated_jobs: bool = False
    dry_run: bool = False
    job_only: bool = True
    max_trials: int = 1
    max_stage: str = "basic"
    trial_store_path: Path = REPO_ROOT / "runs" / "optimization" / "trials.jsonl"
    trial_store_backend: Literal["jsonl", "sqlite"] = "jsonl"
    trial_store_sqlite_path: Path = (
        REPO_ROOT / "runs" / "optimization" / "trials.sqlite"
    )
    objective_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "quality_score": 1.0,
            "peak_vram_mb": -0.00005,
            "latency_ms": -0.001,
        }
    )

    @field_validator("max_generated_jobs", "max_trials")
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("optimization counts must be positive")
        return value

    @field_validator("objective_weights")
    @classmethod
    def _finite_objective_weights(cls, value: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for key, weight in value.items():
            if not key:
                raise ValueError("objective weight names must not be empty")
            if weight != weight or weight in (float("inf"), float("-inf")):
                raise ValueError("objective weights must be finite")
            cleaned[str(key)] = float(weight)
        return cleaned


class LoaderSettings(BaseModel):
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last: bool = False

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

    @field_validator("prefetch_factor")
    @classmethod
    def _optional_positive_prefetch(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("prefetch_factor must be positive when set")
        return value

    @model_validator(mode="after")
    def validate_worker_options(self) -> LoaderSettings:
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")
        if self.num_workers == 0 and self.prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0")
        return self


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
    split_manifest_path: Path | None = None
    dimensions: tuple[str, ...] = ("atomic", "long_sequence")
    categories: tuple[str, ...] = ()
    include_items: tuple[str, ...] = ()
    exclude_items: tuple[str, ...] = ()
    train_items: tuple[str, ...] = ()
    validation_items: tuple[str, ...] = ()
    test_items: tuple[str, ...] = ()
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
        "test_items",
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
        groups = (
            ("train_items", set(self.train_items)),
            ("validation_items", set(self.validation_items)),
            ("test_items", set(self.test_items)),
        )
        for index, (left_name, left_items) in enumerate(groups):
            for right_name, right_items in groups[index + 1 :]:
                overlap = left_items & right_items
                if overlap:
                    joined = ", ".join(sorted(overlap))
                    raise ValueError(f"{left_name} and {right_name} overlap: {joined}")
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
    coordinate_transform: CoordinateTransformSettings = Field(
        default_factory=CoordinateTransformSettings
    )
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    local_encoder: LocalEncoderConfig = Field(default_factory=LocalEncoderConfig)
    global_encoder: GlobalEncoderConfig = Field(default_factory=GlobalEncoderConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    smet: SMETConfig = Field(default_factory=SMETConfig)
    candidate_cache: CandidateCacheSettings = Field(
        default_factory=CandidateCacheSettings
    )
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)
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
    split_manifest_path = data_input.get("split_manifest_path")
    if split_manifest_path is not None:
        path = Path(split_manifest_path).expanduser()
        data_input["split_manifest_path"] = (
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
    candidate_cache = dict(resolved.get("candidate_cache") or {})
    output_root = candidate_cache.get("output_root")
    if output_root is not None:
        path = Path(output_root).expanduser()
        candidate_cache["output_root"] = (
            path if path.is_absolute() else (base_dir / path).resolve()
        )
    resolved["candidate_cache"] = candidate_cache
    optimization = dict(resolved.get("optimization") or {})
    for key in ("trial_store_path", "trial_store_sqlite_path"):
        value = optimization.get(key)
        if value is not None:
            path = Path(value).expanduser()
            optimization[key] = (
                path if path.is_absolute() else (base_dir / path).resolve()
            )
    resolved["optimization"] = optimization
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
    "CandidateCacheSettings",
    "CoordinateTransformSettings",
    "DataSplit",
    "DataInputSettings",
    "EvaluationSettings",
    "FusionConfig",
    "GlobalEncoderConfig",
    "InputSettings",
    "LoaderSettings",
    "LocalEncoderConfig",
    "MemoryConfig",
    "OptimizationSettings",
    "PlayfieldRectSettings",
    "REPO_ROOT",
    "RuntimeSettings",
    "SMETConfig",
    "Settings",
    "SettingsError",
    "TemporalConfig",
    "TemporalLossWeights",
    "TilingConfig",
    "TrainingSettings",
    "VisualizationSettings",
    "load_settings",
]
