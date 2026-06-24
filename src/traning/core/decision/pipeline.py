from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from start.checks import TrainingStartupCheckReport, run_training_startup_checks
from traning.conf import DataSplit, Settings, load_settings
from traning.core.dataset_import import DataInputReport
from traning.core.decision.generator import (
    CandidateCacheBuildResult,
    generate_candidate_cache,
)
from traning.core.decision.runner import (
    TemporalDecisionRunResult,
    run_temporal_decision,
)
from traning.core.spatial import SpatialTrainingResult, run_spatial_training
from traning.core.temporal import TemporalTrainingResult, run_temporal_training


@dataclass(frozen=True)
class TrainingStage:
    key: str
    description: str


@dataclass(frozen=True)
class FullTrainingRunConfig:
    run_dir: Path
    device: torch.device
    split: DataSplit = "train"
    spatial_max_steps: int = 1
    temporal_max_steps: int = 1
    spatial_learning_rate: float = 1e-4
    temporal_learning_rate: float = 1e-4
    patch_limit: int | None = 1
    cache_max_frames: int | None = 1
    max_candidates: int | None = None
    score_threshold: float | None = None
    nms_radius_px: float | None = None
    slider_threshold: float | None = None
    max_slider_paths: int | None = None
    sequence_length: int | None = None
    candidate_slots: int | None = None

    def __post_init__(self) -> None:
        if self.spatial_max_steps <= 0:
            raise ValueError("spatial_max_steps must be positive")
        if self.temporal_max_steps <= 0:
            raise ValueError("temporal_max_steps must be positive")
        if self.spatial_learning_rate <= 0:
            raise ValueError("spatial_learning_rate must be positive")
        if self.temporal_learning_rate <= 0:
            raise ValueError("temporal_learning_rate must be positive")
        for name in (
            "patch_limit",
            "cache_max_frames",
            "max_candidates",
            "max_slider_paths",
            "sequence_length",
            "candidate_slots",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")


@dataclass(frozen=True)
class FullTrainingRunResult:
    run_dir: Path
    summary_path: Path
    startup_checks: TrainingStartupCheckReport
    data_input: DataInputReport
    spatial: SpatialTrainingResult
    candidate_cache: CandidateCacheBuildResult
    temporal: TemporalTrainingResult
    decision: TemporalDecisionRunResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "summary_path": self.summary_path,
            "startup_checks": self.startup_checks.as_dict(),
            "data_input": _data_input_report_dict(self.data_input),
            "spatial": self.spatial.as_dict(),
            "candidate_cache": self.candidate_cache.as_dict(),
            "temporal": self.temporal.as_dict(),
            "decision": self.decision.as_dict(),
        }

    def as_summary(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "summary_path": self.summary_path,
            "startup_checks_ok": self.startup_checks.ok,
            "startup_warnings": len(self.startup_checks.report.warnings),
            "segments": self.data_input.segment_count,
            "estimated_frames": self.data_input.frame_count_estimate,
            "spatial_steps": self.spatial.steps,
            "spatial_loss": self.spatial.last_loss,
            "cache_frames": self.candidate_cache.frames,
            "cache_candidates": self.candidate_cache.candidates,
            "temporal_steps": self.temporal.steps,
            "temporal_loss": self.temporal.final_loss,
            "temporal_checkpoint": self.temporal.checkpoint_path,
            "decision_frames": self.decision.frames,
            "decision_output": self.decision.output_dir,
        }


TRAINING_STAGES = (
    TrainingStage("data_input", "inspect configured training split"),
    TrainingStage("spatial", "train first-version spatial model"),
    TrainingStage("candidate_cache", "build spatial candidate cache"),
    TrainingStage("temporal", "train causal temporal decision model"),
    TrainingStage("decision", "export frame-level temporal decisions"),
)


def run_full_training_pipeline(
    settings: Settings,
    *,
    config: FullTrainingRunConfig,
) -> FullTrainingRunResult:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    startup_checks = run_training_startup_checks(
        settings,
        split=config.split,
        device=config.device,
    )
    startup_checks.raise_for_errors()
    data_report = startup_checks.data_input

    spatial = run_spatial_training(
        settings,
        device=config.device,
        run_dir=config.run_dir / "spatial",
        split=config.split,
        max_steps=config.spatial_max_steps,
        learning_rate=config.spatial_learning_rate,
        patch_limit=config.patch_limit,
    )
    candidate_cache = generate_candidate_cache(
        settings,
        output_dir=config.run_dir / "candidate_cache",
        device=config.device,
        split=config.split,
        max_frames=config.cache_max_frames,
        patch_limit=config.patch_limit,
        max_candidates=config.max_candidates,
        score_threshold=config.score_threshold,
        nms_radius_px=config.nms_radius_px,
        slider_threshold=config.slider_threshold,
        max_slider_paths=config.max_slider_paths,
    )
    temporal = run_temporal_training(
        settings,
        cache_dir=candidate_cache.output_dir,
        device=config.device,
        run_dir=config.run_dir / "temporal",
        max_steps=config.temporal_max_steps,
        learning_rate=config.temporal_learning_rate,
        sequence_length=config.sequence_length,
        candidate_slots=config.candidate_slots,
    )
    decision = run_temporal_decision(
        settings,
        cache_dir=candidate_cache.output_dir,
        checkpoint_path=temporal.checkpoint_path,
        output_dir=config.run_dir / "decision",
        device=config.device,
    )
    summary_path = config.run_dir / "full_training_summary.json"
    result = FullTrainingRunResult(
        run_dir=config.run_dir,
        summary_path=summary_path,
        startup_checks=startup_checks,
        data_input=data_report,
        spatial=spatial,
        candidate_cache=candidate_cache,
        temporal=temporal,
        decision=decision,
    )
    summary_path.write_text(
        json.dumps(_json_ready(result.as_dict()), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return result


def run_pipeline(
    settings: Settings | None = None,
    *,
    config: FullTrainingRunConfig | None = None,
) -> FullTrainingRunResult:
    selected = settings or load_settings()
    selected_config = config or FullTrainingRunConfig(
        run_dir=Path("runs/full_training"),
        device=_device_from_settings(selected),
    )
    return run_full_training_pipeline(selected, config=selected_config)


def _data_input_report_dict(report: DataInputReport) -> dict[str, Any]:
    return {
        "split": report.split,
        "segment_count": report.segment_count,
        "frame_count_estimate": report.frame_count_estimate,
        "item_counts": report.item_counts,
        "category_counts": report.category_counts,
        "dimension_counts": report.dimension_counts,
        "issue_count": report.issue_count,
        "issues": report.issues,
        "ok": report.ok,
    }


def _device_from_settings(settings: Settings) -> torch.device:
    if settings.runtime.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(settings.runtime.device)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    return value


__all__ = [
    "DataInputReport",
    "FullTrainingRunConfig",
    "FullTrainingRunResult",
    "TRAINING_STAGES",
    "TrainingStage",
    "run_full_training_pipeline",
    "run_pipeline",
]
