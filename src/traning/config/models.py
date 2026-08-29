"""OSU V2 的严格配置模型与加载边界。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import TypeVar

import yaml
from package import AffineOsuVideoTransform

from .versions import CANDIDATE_CACHE_SCHEMA_VERSION, TELEMETRY_SCHEMA_VERSION


V2_CONFIG_SCHEMA_VERSION = 1
"""当前唯一受支持的顶层配置版本。"""

AffineMatrix = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]
"""osu 坐标到原始视频帧像素的 2×3 affine 矩阵。"""


class RuntimeDevice(str, Enum):
    """V2 允许使用的计算设备。"""

    CPU = "cpu"
    CUDA = "cuda"


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数")
    if value < minimum:
        raise ValueError(f"{name} 必须大于等于 {minimum}")
    return value


def _require_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} 必须大于等于 {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} 必须小于等于 {maximum}")
    return result


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} 必须是布尔值")
    return value


def _require_path(name: str, value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} 必须是 pathlib.Path")
    rendered = str(value)
    if not rendered or "\x00" in rendered:
        raise ValueError(f"{name} 必须是有效的非空路径")
    return value


def _require_horizons(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise TypeError(f"{name} 必须是非空整数元组")
    checked = tuple(
        _require_int(f"{name}[{index}]", item) for index, item in enumerate(value)
    )
    if checked != tuple(sorted(set(checked))):
        raise ValueError(f"{name} 必须严格递增且不能重复")
    if checked[0] != 0 or not any(horizon > 0 for horizon in checked):
        raise ValueError(f"{name} 必须以 0 开头且至少包含一个正 horizon")
    return checked


def _optional_affine_matrix(name: str, value: object) -> AffineMatrix | None:
    """严格解析 2×3 矩阵，并复用共享 transform 验证可逆性。"""

    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} 必须是 2×3 数值数组或 null")
    rows = tuple(value)
    if len(rows) != 2:
        raise ValueError(f"{name} 必须恰好包含两行")
    converted_rows: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"{name}[{row_index}] 必须是数值数组")
        values = tuple(row)
        if len(values) != 3:
            raise ValueError(f"{name}[{row_index}] 必须恰好包含三个数值")
        converted_rows.append(
            tuple(
                _require_real(f"{name}[{row_index}][{column_index}]", item)
                for column_index, item in enumerate(values)
            )
        )
    matrix: AffineMatrix = (converted_rows[0], converted_rows[1])
    # 由全局稳定 API 校验 determinant，避免配置层复制另一套可逆性公式。
    AffineOsuVideoTransform(matrix)
    return matrix


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    """单帧候选感知配置。"""

    input_channels: int = 3
    frame_width: int = 512
    frame_height: int = 288
    embedding_dim: int = 32
    max_candidates: int = 64
    score_threshold: float = 0.05
    nms_radius_px: float = 24.0
    global_frozen: bool = False
    global_pretrained: bool = False

    def __post_init__(self) -> None:
        _require_int("perception.input_channels", self.input_channels, minimum=1)
        _require_int("perception.frame_width", self.frame_width, minimum=1)
        _require_int("perception.frame_height", self.frame_height, minimum=1)
        _require_int("perception.embedding_dim", self.embedding_dim, minimum=1)
        _require_int("perception.max_candidates", self.max_candidates, minimum=1)
        _require_real(
            "perception.score_threshold", self.score_threshold, minimum=0.0, maximum=1.0
        )
        _require_real("perception.nms_radius_px", self.nms_radius_px, minimum=0.0)
        _require_bool("perception.global_frozen", self.global_frozen)
        _require_bool("perception.global_pretrained", self.global_pretrained)


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """跨帧候选关联和轨迹生命周期配置。"""

    max_distance_px: float = 64.0
    max_embedding_distance: float = 0.5
    max_missed_frames: int = 6
    min_association_confidence: float = 0.05
    spatial_weight: float = 0.55
    embedding_weight: float = 0.35
    type_weight: float = 0.10
    max_total_cost: float = 0.95

    def __post_init__(self) -> None:
        _require_real("tracking.max_distance_px", self.max_distance_px, minimum=0.0)
        if self.max_distance_px <= 0.0:
            raise ValueError("tracking.max_distance_px 必须大于 0")
        _require_real(
            "tracking.max_embedding_distance",
            self.max_embedding_distance,
            minimum=0.0,
            maximum=1.0,
        )
        if self.max_embedding_distance <= 0.0:
            raise ValueError("tracking.max_embedding_distance 必须大于 0")
        _require_int("tracking.max_missed_frames", self.max_missed_frames)
        _require_real(
            "tracking.min_association_confidence",
            self.min_association_confidence,
            minimum=0.0,
            maximum=1.0,
        )
        weights = (
            self.spatial_weight,
            self.embedding_weight,
            self.type_weight,
        )
        for name, value in zip(
            ("spatial_weight", "embedding_weight", "type_weight"),
            weights,
            strict=True,
        ):
            _require_real(f"tracking.{name}", value, minimum=0.0, maximum=1.0)
        if not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("tracking cost weights 的和必须为 1")
        _require_real(
            "tracking.max_total_cost",
            self.max_total_cost,
            minimum=0.0,
            maximum=1.0,
        )


@dataclass(frozen=True, slots=True)
class BeliefConfig:
    """每条轨迹的时序信念模型配置。"""

    input_dim: int = 64
    hidden_dim: int = 128
    layers: int = 1
    max_time_since_seen_ms: int = 500

    def __post_init__(self) -> None:
        _require_int("belief.input_dim", self.input_dim, minimum=1)
        _require_int("belief.hidden_dim", self.hidden_dim, minimum=1)
        _require_int("belief.layers", self.layers, minimum=1)
        _require_int(
            "belief.max_time_since_seen_ms", self.max_time_since_seen_ms, minimum=1
        )


@dataclass(frozen=True, slots=True)
class OutcomeConfig:
    """点击结果分布模型配置。"""

    hidden_dims: tuple[int, ...] = (128, 64)
    category_count: int = 5
    horizons_ms: tuple[int, ...] = (0, 16, 32, 48, 64)
    calibration_bins: int = 15

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_dims, tuple) or not self.hidden_dims:
            raise TypeError("outcome.hidden_dims 必须是非空整数元组")
        for index, dimension in enumerate(self.hidden_dims):
            _require_int(f"outcome.hidden_dims[{index}]", dimension, minimum=1)
        _require_int("outcome.category_count", self.category_count, minimum=2)
        if self.category_count != 5:
            raise ValueError("outcome.category_count 必须与 canonical 五分类契约一致")
        _require_horizons("outcome.horizons_ms", self.horizons_ms)
        _require_int("outcome.calibration_bins", self.calibration_bins, minimum=2)


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    """基于未来价值的最优停止决策配置。"""

    horizons_ms: tuple[int, ...] = (0, 16, 32, 48, 64)
    click_cost: float = 0.0
    invalid_penalty: float = 1.0
    miss_penalty: float = 0.75
    expire_penalty: float = 0.5
    min_confidence: float = 0.0
    risk_lambda: float = 0.1
    wait_cost: float = 0.0

    def __post_init__(self) -> None:
        _require_horizons("decision.horizons_ms", self.horizons_ms)
        _require_real("decision.click_cost", self.click_cost, minimum=0.0)
        _require_real("decision.invalid_penalty", self.invalid_penalty, minimum=0.0)
        _require_real("decision.miss_penalty", self.miss_penalty, minimum=0.0)
        _require_real("decision.expire_penalty", self.expire_penalty, minimum=0.0)
        _require_real(
            "decision.min_confidence", self.min_confidence, minimum=0.0, maximum=1.0
        )
        _require_real("decision.risk_lambda", self.risk_lambda, minimum=0.0)
        _require_real("decision.wait_cost", self.wait_cost, minimum=0.0)


@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    """typed 样本 DataLoader 的进程与锁页内存配置。"""

    workers: int = 0
    pin_memory: bool = False

    def __post_init__(self) -> None:
        _require_int("data.loader.workers", self.workers, minimum=0)
        _require_bool("data.loader.pin_memory", self.pin_memory)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """segment 数据发现、item 划分、确定性取帧与加载配置。"""

    seed: int = 0
    dataset_root: Path = Path("training_package/video_segments")
    split_manifest: Path = Path(
        "training_package/splits/dataset_split_manifest.json"
    )
    sample_fps: float = 60.0
    frame_step: int = 1
    max_segments_per_split: int | None = None
    max_frames_per_segment: int | None = None
    visibility_post_ms: float = 100.0
    loader: DataLoaderConfig = DataLoaderConfig()

    def __post_init__(self) -> None:
        _require_int("data.seed", self.seed)
        _require_path("data.dataset_root", self.dataset_root)
        _require_path("data.split_manifest", self.split_manifest)
        _require_real("data.sample_fps", self.sample_fps, minimum=0.0)
        if self.sample_fps <= 0.0:
            raise ValueError("data.sample_fps 必须大于 0")
        _require_int("data.frame_step", self.frame_step, minimum=1)
        for field_name, value in (
            ("max_segments_per_split", self.max_segments_per_split),
            ("max_frames_per_segment", self.max_frames_per_segment),
        ):
            if value is not None:
                _require_int(f"data.{field_name}", value, minimum=1)
        _require_real(
            "data.visibility_post_ms",
            self.visibility_post_ms,
            minimum=0.0,
        )
        if not isinstance(self.loader, DataLoaderConfig):
            raise TypeError("data.loader 必须是 DataLoaderConfig")


@dataclass(frozen=True, slots=True)
class CoordinateConfig:
    """与 affine 标定绑定的原视频尺寸、方程及可选审计证据。"""

    source_width: int = 1484
    source_height: int = 846
    transform_identity: str = "unconfigured"
    affine_matrix: AffineMatrix | None = None
    calibration_evidence_path: Path | None = None

    def __post_init__(self) -> None:
        _require_int("coordinates.source_width", self.source_width, minimum=1)
        _require_int("coordinates.source_height", self.source_height, minimum=1)
        if (
            not isinstance(self.transform_identity, str)
            or not self.transform_identity
            or self.transform_identity != self.transform_identity.strip()
        ):
            raise ValueError("coordinates.transform_identity 必须非空且无首尾空格")
        if self.affine_matrix is not None and self.transform_identity == "unconfigured":
            raise ValueError("配置 affine_matrix 时必须声明 transform_identity")
        _optional_affine_matrix("coordinates.affine_matrix", self.affine_matrix)
        if self.calibration_evidence_path is not None:
            _require_path(
                "coordinates.calibration_evidence_path",
                self.calibration_evidence_path,
            )
            if self.affine_matrix is None:
                raise ValueError(
                    "配置 calibration_evidence_path 时必须提供 affine_matrix"
                )


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """候选缓存仓库配置。"""

    schema_version: int = CANDIDATE_CACHE_SCHEMA_VERSION
    directory: Path = Path(".cache/traning")

    def __post_init__(self) -> None:
        if (
            _require_int("cache.schema_version", self.schema_version, minimum=1)
            != CANDIDATE_CACHE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"cache.schema_version 仅支持 {CANDIDATE_CACHE_SCHEMA_VERSION}"
            )
        _require_path("cache.directory", self.directory)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """设备与数值运行时配置。"""

    device: RuntimeDevice = RuntimeDevice.CUDA
    require_cuda: bool = True
    amp: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.device, RuntimeDevice):
            raise TypeError("runtime.device 必须是 RuntimeDevice")
        _require_bool("runtime.require_cuda", self.require_cuda)
        _require_bool("runtime.amp", self.amp)
        if self.require_cuda and self.device is not RuntimeDevice.CUDA:
            raise ValueError("runtime.require_cuda=True 时 device 必须为 cuda")
        if self.amp and self.device is not RuntimeDevice.CUDA:
            raise ValueError("runtime.amp=True 时 device 必须为 cuda")


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """只读遥测事件存储配置。"""

    schema_version: int = TELEMETRY_SCHEMA_VERSION
    directory: Path = Path("output/traning/telemetry")

    def __post_init__(self) -> None:
        if (
            _require_int("telemetry.schema_version", self.schema_version, minimum=1)
            != TELEMETRY_SCHEMA_VERSION
        ):
            raise ValueError(
                f"telemetry.schema_version 仅支持 {TELEMETRY_SCHEMA_VERSION}"
            )
        _require_path("telemetry.directory", self.directory)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """V2 各训练阶段共享的优化配置。"""

    seed: int = 0
    batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        _require_int("training.seed", self.seed)
        _require_int("training.batch_size", self.batch_size, minimum=1)
        _require_int("training.epochs", self.epochs, minimum=1)
        _require_real("training.learning_rate", self.learning_rate, minimum=0.0)
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate 必须大于 0")
        _require_real("training.weight_decay", self.weight_decay, minimum=0.0)


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """严格验收参数搜索的停止预算；None 表示不设 trial 数上限。"""

    max_trials: int | None = None

    def __post_init__(self) -> None:
        if self.max_trials is not None:
            _require_int("optimization.max_trials", self.max_trials, minimum=1)


@dataclass(frozen=True, slots=True)
class V2Config:
    """OSU V2 的单一顶层配置。"""

    schema_version: int = V2_CONFIG_SCHEMA_VERSION
    perception: PerceptionConfig = PerceptionConfig()
    tracking: TrackingConfig = TrackingConfig()
    belief: BeliefConfig = BeliefConfig()
    outcome: OutcomeConfig = OutcomeConfig()
    decision: DecisionConfig = DecisionConfig()
    data: DataConfig = DataConfig()
    coordinates: CoordinateConfig = CoordinateConfig()
    cache: CacheConfig = CacheConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    training: TrainingConfig = TrainingConfig()
    optimization: OptimizationConfig = OptimizationConfig()

    def __post_init__(self) -> None:
        if (
            _require_int("schema_version", self.schema_version, minimum=1)
            != V2_CONFIG_SCHEMA_VERSION
        ):
            raise ValueError(
                f"不支持 schema_version={self.schema_version}；"
                f"仅支持 {V2_CONFIG_SCHEMA_VERSION}"
            )
        section_types = (
            ("perception", self.perception, PerceptionConfig),
            ("tracking", self.tracking, TrackingConfig),
            ("belief", self.belief, BeliefConfig),
            ("outcome", self.outcome, OutcomeConfig),
            ("decision", self.decision, DecisionConfig),
            ("data", self.data, DataConfig),
            ("coordinates", self.coordinates, CoordinateConfig),
            ("cache", self.cache, CacheConfig),
            ("runtime", self.runtime, RuntimeConfig),
            ("telemetry", self.telemetry, TelemetryConfig),
            ("training", self.training, TrainingConfig),
            ("optimization", self.optimization, OptimizationConfig),
        )
        for name, section, expected_type in section_types:
            if not isinstance(section, expected_type):
                raise TypeError(f"{name} 必须是 {expected_type.__name__}")
        if self.outcome.horizons_ms != self.decision.horizons_ms:
            raise ValueError("outcome.horizons_ms 必须与 decision.horizons_ms 完全一致")
        if self.data.seed != self.training.seed:
            raise ValueError("data.seed 必须与 training.seed 一致")
        if self.cache.directory == self.telemetry.directory:
            raise ValueError("cache.directory 与 telemetry.directory 不能相同")
        protected_directories = {self.cache.directory, self.telemetry.directory}
        if self.data.dataset_root in protected_directories:
            raise ValueError("data.dataset_root 不能指向缓存或遥测目录")
        if self.data.split_manifest in protected_directories:
            raise ValueError("data.split_manifest 不能指向缓存或遥测目录")


_T = TypeVar("_T")


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是映射")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} 的键必须是字符串")
    return value


def _section(
    parent: Mapping[str, object],
    name: str,
    model_type: type[_T],
) -> Mapping[str, object]:
    raw = parent.get(name, {})
    section = _mapping(name, raw)
    allowed = {field.name for field in fields(model_type)}
    unknown = set(section) - allowed
    if unknown:
        rendered = ", ".join(sorted(unknown))
        raise ValueError(f"{name} 含未知键: {rendered}")
    return section


def _value(section: Mapping[str, object], name: str, default: _T) -> object | _T:
    return section[name] if name in section else default


def _path_value(name: str, value: object) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"{name} 必须是路径字符串")
    if not value or "\x00" in value:
        raise ValueError(f"{name} 必须是有效的非空路径字符串")
    return Path(value)


def _optional_path_value(name: str, value: object) -> Path | None:
    if value is None:
        return None
    return _path_value(name, value)


def _tuple_of_ints(name: str, value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} 必须是整数数组")
    return tuple(
        _require_int(f"{name}[{index}]", item) for index, item in enumerate(value)
    )


def _optional_positive_int(name: str, value: object) -> int | None:
    """严格解析可选正整数，不把布尔值或字符串当作上限。"""

    if value is None:
        return None
    return _require_int(name, value, minimum=1)


def _parse_perception(root: Mapping[str, object]) -> PerceptionConfig:
    raw = _section(root, "perception", PerceptionConfig)
    defaults = PerceptionConfig()
    return PerceptionConfig(
        input_channels=_require_int(
            "perception.input_channels",
            _value(raw, "input_channels", defaults.input_channels),
            minimum=1,
        ),
        frame_width=_require_int(
            "perception.frame_width",
            _value(raw, "frame_width", defaults.frame_width),
            minimum=1,
        ),
        frame_height=_require_int(
            "perception.frame_height",
            _value(raw, "frame_height", defaults.frame_height),
            minimum=1,
        ),
        embedding_dim=_require_int(
            "perception.embedding_dim",
            _value(raw, "embedding_dim", defaults.embedding_dim),
            minimum=1,
        ),
        max_candidates=_require_int(
            "perception.max_candidates",
            _value(raw, "max_candidates", defaults.max_candidates),
            minimum=1,
        ),
        score_threshold=_require_real(
            "perception.score_threshold",
            _value(raw, "score_threshold", defaults.score_threshold),
            minimum=0.0,
            maximum=1.0,
        ),
        nms_radius_px=_require_real(
            "perception.nms_radius_px",
            _value(raw, "nms_radius_px", defaults.nms_radius_px),
            minimum=0.0,
        ),
        global_frozen=_require_bool(
            "perception.global_frozen",
            _value(raw, "global_frozen", defaults.global_frozen),
        ),
        global_pretrained=_require_bool(
            "perception.global_pretrained",
            _value(raw, "global_pretrained", defaults.global_pretrained),
        ),
    )


def _parse_tracking(root: Mapping[str, object]) -> TrackingConfig:
    raw = _section(root, "tracking", TrackingConfig)
    defaults = TrackingConfig()
    return TrackingConfig(
        max_distance_px=_require_real(
            "tracking.max_distance_px",
            _value(raw, "max_distance_px", defaults.max_distance_px),
            minimum=0.0,
        ),
        max_embedding_distance=_require_real(
            "tracking.max_embedding_distance",
            _value(raw, "max_embedding_distance", defaults.max_embedding_distance),
            minimum=0.0,
            maximum=1.0,
        ),
        max_missed_frames=_require_int(
            "tracking.max_missed_frames",
            _value(raw, "max_missed_frames", defaults.max_missed_frames),
        ),
        min_association_confidence=_require_real(
            "tracking.min_association_confidence",
            _value(
                raw, "min_association_confidence", defaults.min_association_confidence
            ),
            minimum=0.0,
            maximum=1.0,
        ),
        spatial_weight=_require_real(
            "tracking.spatial_weight",
            _value(raw, "spatial_weight", defaults.spatial_weight),
            minimum=0.0,
            maximum=1.0,
        ),
        embedding_weight=_require_real(
            "tracking.embedding_weight",
            _value(raw, "embedding_weight", defaults.embedding_weight),
            minimum=0.0,
            maximum=1.0,
        ),
        type_weight=_require_real(
            "tracking.type_weight",
            _value(raw, "type_weight", defaults.type_weight),
            minimum=0.0,
            maximum=1.0,
        ),
        max_total_cost=_require_real(
            "tracking.max_total_cost",
            _value(raw, "max_total_cost", defaults.max_total_cost),
            minimum=0.0,
            maximum=1.0,
        ),
    )


def _parse_belief(root: Mapping[str, object]) -> BeliefConfig:
    raw = _section(root, "belief", BeliefConfig)
    defaults = BeliefConfig()
    return BeliefConfig(
        input_dim=_require_int(
            "belief.input_dim", _value(raw, "input_dim", defaults.input_dim), minimum=1
        ),
        hidden_dim=_require_int(
            "belief.hidden_dim",
            _value(raw, "hidden_dim", defaults.hidden_dim),
            minimum=1,
        ),
        layers=_require_int(
            "belief.layers", _value(raw, "layers", defaults.layers), minimum=1
        ),
        max_time_since_seen_ms=_require_int(
            "belief.max_time_since_seen_ms",
            _value(raw, "max_time_since_seen_ms", defaults.max_time_since_seen_ms),
            minimum=1,
        ),
    )


def _parse_outcome(root: Mapping[str, object]) -> OutcomeConfig:
    raw = _section(root, "outcome", OutcomeConfig)
    defaults = OutcomeConfig()
    return OutcomeConfig(
        hidden_dims=_tuple_of_ints(
            "outcome.hidden_dims", _value(raw, "hidden_dims", defaults.hidden_dims)
        ),
        category_count=_require_int(
            "outcome.category_count",
            _value(raw, "category_count", defaults.category_count),
            minimum=2,
        ),
        horizons_ms=_tuple_of_ints(
            "outcome.horizons_ms", _value(raw, "horizons_ms", defaults.horizons_ms)
        ),
        calibration_bins=_require_int(
            "outcome.calibration_bins",
            _value(raw, "calibration_bins", defaults.calibration_bins),
            minimum=2,
        ),
    )


def _parse_decision(root: Mapping[str, object]) -> DecisionConfig:
    raw = _section(root, "decision", DecisionConfig)
    defaults = DecisionConfig()
    return DecisionConfig(
        horizons_ms=_tuple_of_ints(
            "decision.horizons_ms", _value(raw, "horizons_ms", defaults.horizons_ms)
        ),
        click_cost=_require_real(
            "decision.click_cost",
            _value(raw, "click_cost", defaults.click_cost),
            minimum=0.0,
        ),
        invalid_penalty=_require_real(
            "decision.invalid_penalty",
            _value(raw, "invalid_penalty", defaults.invalid_penalty),
            minimum=0.0,
        ),
        miss_penalty=_require_real(
            "decision.miss_penalty",
            _value(raw, "miss_penalty", defaults.miss_penalty),
            minimum=0.0,
        ),
        expire_penalty=_require_real(
            "decision.expire_penalty",
            _value(raw, "expire_penalty", defaults.expire_penalty),
            minimum=0.0,
        ),
        min_confidence=_require_real(
            "decision.min_confidence",
            _value(raw, "min_confidence", defaults.min_confidence),
            minimum=0.0,
            maximum=1.0,
        ),
        risk_lambda=_require_real(
            "decision.risk_lambda",
            _value(raw, "risk_lambda", defaults.risk_lambda),
            minimum=0.0,
        ),
        wait_cost=_require_real(
            "decision.wait_cost",
            _value(raw, "wait_cost", defaults.wait_cost),
            minimum=0.0,
        ),
    )


def _parse_data(root: Mapping[str, object]) -> DataConfig:
    raw = _section(root, "data", DataConfig)
    defaults = DataConfig()
    loader_raw = _section(raw, "loader", DataLoaderConfig)
    loader_defaults = DataLoaderConfig()
    return DataConfig(
        seed=_require_int("data.seed", _value(raw, "seed", defaults.seed)),
        dataset_root=_path_value(
            "data.dataset_root",
            _value(raw, "dataset_root", str(defaults.dataset_root)),
        ),
        split_manifest=_path_value(
            "data.split_manifest",
            _value(raw, "split_manifest", str(defaults.split_manifest)),
        ),
        sample_fps=_require_real(
            "data.sample_fps",
            _value(raw, "sample_fps", defaults.sample_fps),
            minimum=0.0,
        ),
        frame_step=_require_int(
            "data.frame_step",
            _value(raw, "frame_step", defaults.frame_step),
            minimum=1,
        ),
        max_segments_per_split=_optional_positive_int(
            "data.max_segments_per_split",
            _value(
                raw,
                "max_segments_per_split",
                defaults.max_segments_per_split,
            ),
        ),
        max_frames_per_segment=_optional_positive_int(
            "data.max_frames_per_segment",
            _value(
                raw,
                "max_frames_per_segment",
                defaults.max_frames_per_segment,
            ),
        ),
        visibility_post_ms=_require_real(
            "data.visibility_post_ms",
            _value(raw, "visibility_post_ms", defaults.visibility_post_ms),
            minimum=0.0,
        ),
        loader=DataLoaderConfig(
            workers=_require_int(
                "data.loader.workers",
                _value(loader_raw, "workers", loader_defaults.workers),
                minimum=0,
            ),
            pin_memory=_require_bool(
                "data.loader.pin_memory",
                _value(
                    loader_raw,
                    "pin_memory",
                    loader_defaults.pin_memory,
                ),
            ),
        ),
    )


def _parse_coordinates(root: Mapping[str, object]) -> CoordinateConfig:
    """解析与原帧尺寸强绑定的可选 affine 标定。"""

    raw = _section(root, "coordinates", CoordinateConfig)
    defaults = CoordinateConfig()
    return CoordinateConfig(
        source_width=_require_int(
            "coordinates.source_width",
            _value(raw, "source_width", defaults.source_width),
            minimum=1,
        ),
        source_height=_require_int(
            "coordinates.source_height",
            _value(raw, "source_height", defaults.source_height),
            minimum=1,
        ),
        transform_identity=_value(
            raw,
            "transform_identity",
            defaults.transform_identity,
        ),
        affine_matrix=_optional_affine_matrix(
            "coordinates.affine_matrix",
            _value(raw, "affine_matrix", defaults.affine_matrix),
        ),
        calibration_evidence_path=_optional_path_value(
            "coordinates.calibration_evidence_path",
            _value(
                raw,
                "calibration_evidence_path",
                defaults.calibration_evidence_path,
            ),
        ),
    )


def _parse_cache(root: Mapping[str, object]) -> CacheConfig:
    raw = _section(root, "cache", CacheConfig)
    defaults = CacheConfig()
    return CacheConfig(
        schema_version=_require_int(
            "cache.schema_version",
            _value(raw, "schema_version", defaults.schema_version),
            minimum=1,
        ),
        directory=_path_value(
            "cache.directory", _value(raw, "directory", str(defaults.directory))
        ),
    )


def _parse_runtime(root: Mapping[str, object]) -> RuntimeConfig:
    raw = _section(root, "runtime", RuntimeConfig)
    defaults = RuntimeConfig()
    device_raw = _value(raw, "device", defaults.device.value)
    if not isinstance(device_raw, str):
        raise TypeError("runtime.device 必须是字符串")
    try:
        device = RuntimeDevice(device_raw)
    except ValueError as error:
        allowed = ", ".join(item.value for item in RuntimeDevice)
        raise ValueError(f"runtime.device 必须是以下值之一: {allowed}") from error
    return RuntimeConfig(
        device=device,
        require_cuda=_require_bool(
            "runtime.require_cuda", _value(raw, "require_cuda", defaults.require_cuda)
        ),
        amp=_require_bool("runtime.amp", _value(raw, "amp", defaults.amp)),
    )


def _parse_telemetry(root: Mapping[str, object]) -> TelemetryConfig:
    raw = _section(root, "telemetry", TelemetryConfig)
    defaults = TelemetryConfig()
    return TelemetryConfig(
        schema_version=_require_int(
            "telemetry.schema_version",
            _value(raw, "schema_version", defaults.schema_version),
            minimum=1,
        ),
        directory=_path_value(
            "telemetry.directory", _value(raw, "directory", str(defaults.directory))
        ),
    )


def _parse_training(root: Mapping[str, object]) -> TrainingConfig:
    raw = _section(root, "training", TrainingConfig)
    defaults = TrainingConfig()
    return TrainingConfig(
        seed=_require_int("training.seed", _value(raw, "seed", defaults.seed)),
        batch_size=_require_int(
            "training.batch_size",
            _value(raw, "batch_size", defaults.batch_size),
            minimum=1,
        ),
        epochs=_require_int(
            "training.epochs", _value(raw, "epochs", defaults.epochs), minimum=1
        ),
        learning_rate=_require_real(
            "training.learning_rate",
            _value(raw, "learning_rate", defaults.learning_rate),
            minimum=0.0,
        ),
        weight_decay=_require_real(
            "training.weight_decay",
            _value(raw, "weight_decay", defaults.weight_decay),
            minimum=0.0,
        ),
    )


def _parse_optimization(root: Mapping[str, object]) -> OptimizationConfig:
    """解析显式 trial 上限；null 保持“未全通过就继续”的默认语义。"""

    raw = _section(root, "optimization", OptimizationConfig)
    defaults = OptimizationConfig()
    max_trials_raw = _value(raw, "max_trials", defaults.max_trials)
    max_trials = (
        None
        if max_trials_raw is None
        else _require_int("optimization.max_trials", max_trials_raw, minimum=1)
    )
    return OptimizationConfig(max_trials=max_trials)


def _load_path(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在或不是文件: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    else:
        raise ValueError("配置文件扩展名必须是 .json、.yaml 或 .yml")
    return _mapping("配置根", raw)


def load_v2_config(source: Mapping[str, object] | Path) -> V2Config:
    """从严格映射或 JSON/YAML 文件加载 V2 配置。

    加载边界不做版本迁移，也不容忍未知键、字符串化数值或无版本输入。
    """

    if isinstance(source, Path):
        root = _load_path(source)
    elif isinstance(source, Mapping):
        root = _mapping("配置根", source)
    else:
        raise TypeError("source 必须是 Mapping[str, object] 或 pathlib.Path")

    allowed = {field.name for field in fields(V2Config)}
    unknown = set(root) - allowed
    if unknown:
        rendered = ", ".join(sorted(unknown))
        raise ValueError(f"配置根含未知键: {rendered}")
    if "schema_version" not in root:
        raise ValueError("配置根缺少 schema_version")
    schema_version = _require_int("schema_version", root["schema_version"], minimum=1)
    if schema_version != V2_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"不支持 schema_version={schema_version}；仅支持 {V2_CONFIG_SCHEMA_VERSION}"
        )

    return V2Config(
        schema_version=schema_version,
        perception=_parse_perception(root),
        tracking=_parse_tracking(root),
        belief=_parse_belief(root),
        outcome=_parse_outcome(root),
        decision=_parse_decision(root),
        data=_parse_data(root),
        coordinates=_parse_coordinates(root),
        cache=_parse_cache(root),
        runtime=_parse_runtime(root),
        telemetry=_parse_telemetry(root),
        training=_parse_training(root),
        optimization=_parse_optimization(root),
    )


def v2_config_to_dict(config: V2Config) -> dict[str, object]:
    """把已验证配置转换为可直接交给 JSON 编码器的字典。"""

    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    return {
        "schema_version": config.schema_version,
        "perception": {
            field.name: getattr(config.perception, field.name)
            for field in fields(PerceptionConfig)
        },
        "tracking": {
            field.name: getattr(config.tracking, field.name)
            for field in fields(TrackingConfig)
        },
        "belief": {
            field.name: getattr(config.belief, field.name)
            for field in fields(BeliefConfig)
        },
        "outcome": {
            "hidden_dims": list(config.outcome.hidden_dims),
            "category_count": config.outcome.category_count,
            "horizons_ms": list(config.outcome.horizons_ms),
            "calibration_bins": config.outcome.calibration_bins,
        },
        "decision": {
            "horizons_ms": list(config.decision.horizons_ms),
            "click_cost": config.decision.click_cost,
            "invalid_penalty": config.decision.invalid_penalty,
            "miss_penalty": config.decision.miss_penalty,
            "expire_penalty": config.decision.expire_penalty,
            "min_confidence": config.decision.min_confidence,
            "risk_lambda": config.decision.risk_lambda,
            "wait_cost": config.decision.wait_cost,
        },
        "data": {
            "seed": config.data.seed,
            "dataset_root": str(config.data.dataset_root),
            "split_manifest": str(config.data.split_manifest),
            "sample_fps": config.data.sample_fps,
            "frame_step": config.data.frame_step,
            "max_segments_per_split": config.data.max_segments_per_split,
            "max_frames_per_segment": config.data.max_frames_per_segment,
            "visibility_post_ms": config.data.visibility_post_ms,
            "loader": {
                "workers": config.data.loader.workers,
                "pin_memory": config.data.loader.pin_memory,
            },
        },
        "coordinates": {
            "source_width": config.coordinates.source_width,
            "source_height": config.coordinates.source_height,
            "transform_identity": config.coordinates.transform_identity,
            "affine_matrix": (
                None
                if config.coordinates.affine_matrix is None
                else [list(row) for row in config.coordinates.affine_matrix]
            ),
            "calibration_evidence_path": (
                str(config.coordinates.calibration_evidence_path)
                if config.coordinates.calibration_evidence_path is not None
                else None
            ),
        },
        "cache": {
            "schema_version": config.cache.schema_version,
            "directory": str(config.cache.directory),
        },
        "runtime": {
            "device": config.runtime.device.value,
            "require_cuda": config.runtime.require_cuda,
            "amp": config.runtime.amp,
        },
        "telemetry": {
            "schema_version": config.telemetry.schema_version,
            "directory": str(config.telemetry.directory),
        },
        "training": {
            field.name: getattr(config.training, field.name)
            for field in fields(TrainingConfig)
        },
        "optimization": {"max_trials": config.optimization.max_trials},
    }
