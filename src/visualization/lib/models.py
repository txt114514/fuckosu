from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


class PipelinePhase(StrEnum):
    STARTUP = "startup"
    DATA_PREPARATION = "data_preparation"
    PRETRAIN_CHECK = "pretrain_check"
    PROGRESSIVE_PREPARATION = "progressive_preparation"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TrainingEvent:
    timestamp: str
    event_type: str
    severity: str
    message_key: str
    message_args: dict[str, object] = field(default_factory=dict)
    phase: str | None = None
    level: str | None = None
    trial_id: str | None = None
    raw_message: str | None = None

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        severity: str,
        message_key: str,
        message_args: dict[str, object] | None = None,
        phase: str | None = None,
        level: str | None = None,
        trial_id: str | None = None,
        raw_message: str | None = None,
    ) -> TrainingEvent:
        return cls(
            timestamp=utc_now(),
            event_type=event_type,
            severity=severity,
            message_key=message_key,
            message_args=message_args or {},
            phase=phase,
            level=level,
            trial_id=trial_id,
            raw_message=raw_message,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineStageState:
    stage_id: str
    name: str
    status: str = "pending"
    started_at: str | None = None
    ended_at: str | None = None
    processed: int = 0
    total: int | None = None
    output_path: str | None = None
    warning_count: int = 0
    error_reason: str | None = None
    blocks_training: bool = False
    message: str | None = None
    score: float | None = None
    threshold: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BestParameterRecord:
    trial_id: str | None = None
    parameters: dict[str, object] = field(default_factory=dict)
    score: float | None = None
    step: int | None = None
    epoch: int | None = None
    level: str | None = None
    grade: str | None = None
    checkpoint_path: str | None = None
    scored_at: str | None = None
    restorable: bool = False
    resolved_config_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CurrentTrainingMetrics:
    loss: float | None = None
    moving_average_loss: float | None = None
    validation_loss: float | None = None
    score: float | None = None
    parameter_best_score: float | None = None
    level_best_score: float | None = None
    run_global_best_score: float | None = None
    inherited_best_score: float | None = None
    gradient_norm: float | None = None
    learning_rate: float | None = None
    clipped: bool | None = None
    overflow: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetUsageState:
    total_segments: int = 0
    sampled_segments: int = 0
    unique_segments: int = 0
    total_frames: int = 0
    sampled_frames: int = 0
    unique_frames: int = 0
    trained_frames: int = 0
    cached_frames: int = 0
    dropped_frames: int = 0
    generated_patches: int = 0
    generated_candidates: int = 0
    used_sequences: int = 0
    duplicate_samples: int = 0
    current_segment: str | None = None
    epoch_complete: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceState:
    gpu_index: int | None = None
    gpu_name: str | None = None
    gpu_allocated_gb: float | None = None
    gpu_reserved_gb: float | None = None
    gpu_peak_allocated_gb: float | None = None
    gpu_peak_reserved_gb: float | None = None
    gpu_total_gb: float | None = None
    gpu_memory_used_gb: float | None = None
    gpu_utilization: float | None = None
    gpu_utilization_avg: float | None = None
    gpu_utilization_max: float | None = None
    gpu_memory_utilization: float | None = None
    gpu_temperature_c: float | None = None
    gpu_power_w: float | None = None
    gpu_monitor_source: str | None = None
    gpu_monitor_error: str | None = None
    cpu_percent: float | None = None
    system_memory_gb: float | None = None
    process_memory_gb: float | None = None
    disk_free_gb: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingStopState:
    reason: str
    message: str
    exit_code: int
    step: int | None = None
    target_step: int | None = None
    latest_checkpoint: str | None = None
    inheritance_path: str | None = None
    saved_at: str = field(default_factory=utc_now)
    details: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingInheritanceSummary:
    path: str
    status: str
    policy: str
    loaded_checkpoint: str | None = None
    downgrade_reasons: tuple[str, ...] = ()
    compatible: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GallerySelectionRequest:
    batch_id: str
    samples_per_group: int
    random_seed: int | None = None


@dataclass
class GalleryRenderRequest:
    sample_key: str
    frame_index: int
    output_path: Path


@dataclass
class GalleryExportRequest:
    output_root: Path
    samples_per_group: int


@dataclass
class TrainingDashboardState:
    run_id: str
    phase: str = "初始化"
    pipeline_phase: str = PipelinePhase.STARTUP.value
    status: str = "pending"
    pipeline_stages: dict[str, PipelineStageState] = field(default_factory=dict)
    current_level: str | None = None
    completed_levels: int = 0
    total_levels: int = 0
    epoch: int = 0
    total_epochs: int | None = None
    global_step: int = 0
    target_global_steps: int | None = None
    spatial_step: int = 0
    spatial_target: int | None = None
    temporal_step: int = 0
    temporal_target: int | None = None
    current_trial_id: str | None = None
    current_parameters: dict[str, object] = field(default_factory=dict)
    current_parameter_status: dict[str, object] = field(default_factory=dict)
    trial_status: str | None = None
    prune_reason: str | None = None
    metrics: CurrentTrainingMetrics = field(default_factory=CurrentTrainingMetrics)
    current_grade: str | None = None
    best_grade: str | None = None
    promotion_status: str | None = "未评级"
    consecutive_passes: int = 0
    required_passes: int | None = None
    category_scores: dict[str, float] = field(default_factory=dict)
    dataset_usage: DatasetUsageState = field(default_factory=DatasetUsageState)
    resources: ResourceState = field(default_factory=ResourceState)
    frames_per_second: float | None = None
    steps_per_second: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    latest_checkpoint: str | None = None
    best_checkpoint: str | None = None
    inheritance_path: str | None = None
    best_parameters: BestParameterRecord = field(default_factory=BestParameterRecord)
    stop_state: TrainingStopState | None = None
    recent_events: list[TrainingEvent] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["pipeline_stages"] = {
            key: stage.as_dict() for key, stage in self.pipeline_stages.items()
        }
        value["recent_events"] = [event.as_dict() for event in self.recent_events]
        return value
