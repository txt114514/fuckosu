"""按递增预算运行训练课程门禁，并在达标后衔接完整训练。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
import yaml

from environment import collect_environment_report
from traning.conf import load_settings
from traning.core.dataset_import import inspect_data_input
from traning.core.decision import (
    CANDIDATE_CACHE_VERSION,
    FullTrainingRunConfig,
    run_full_training_pipeline,
)
from traning.core.model_export import (
    ModelArtifactSpec,
    export_model_artifact,
    smoke_test_model_artifact,
    validate_model_artifact,
)
from traning.core.training_inheritance import TrainingPosition
from traning.core.optimization import AGGREGATE_SCORE_VERSION, training_job_from_dict
from traning.state import CurriculumStage
from traning.state.versioning import collect_code_version
from visualization.lib import (
    DatasetUsageState,
    PipelinePhase,
    PipelineStageState,
    TrainingEvent,
    TrainingReporter,
    TrainingStopState,
    collect_resource_state,
    create_dashboard_reporter,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts") / "training_ramp"
DEFAULT_FULL_CONFIG = Path("configs") / "model_full_small_vram.yaml"


@dataclass(frozen=True)
class RampLevelSpec:
    key: str
    label: str
    spatial_steps: int
    temporal_steps: int
    patch_limit: int
    cache_frames: int
    sequence_length: int
    candidate_slots: int
    gallery_samples_per_group: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "spatial_steps": self.spatial_steps,
            "temporal_steps": self.temporal_steps,
            "patch_limit": self.patch_limit,
            "cache_frames": self.cache_frames,
            "sequence_length": self.sequence_length,
            "candidate_slots": self.candidate_slots,
            "gallery_samples_per_group": self.gallery_samples_per_group,
        }


@dataclass(frozen=True)
class RampTarget:
    spatial_steps: int = 600
    temporal_steps: int = 600
    patch_limit: int = 4
    cache_frames: int = 3000
    sequence_length: int = 96
    candidate_slots: int = 24
    gallery_samples_per_group: int = 4

    def as_level(self) -> RampLevelSpec:
        return RampLevelSpec(
            key="target",
            label="target_readiness",
            spatial_steps=self.spatial_steps,
            temporal_steps=self.temporal_steps,
            patch_limit=self.patch_limit,
            cache_frames=self.cache_frames,
            sequence_length=self.sequence_length,
            candidate_slots=self.candidate_slots,
            gallery_samples_per_group=self.gallery_samples_per_group,
        )


@dataclass(frozen=True)
class RampRunResult:
    output_dir: Path
    manifest_path: Path
    final_readiness_path: Path
    completed_level: str
    full_training_started: bool
    full_training_run_dir: Path | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "final_readiness_path": self.final_readiness_path,
            "completed_level": self.completed_level,
            "full_training_started": self.full_training_started,
            "full_training_run_dir": self.full_training_run_dir,
            "status": self.status,
        }


class RampGateError(RuntimeError):
    pass


class RampEvaluationGateError(RampGateError):
    """表示训练产物有效，但当前参数尚未通过严格评估。"""


class RampSearchExhausted(RampGateError):
    """表示仍有下一轮参数，但自动执行被关闭或试验预算已耗尽。"""


def run_training_ramp(
    *,
    config_path: Path,
    device: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    target_config_path: Path | None = None,
    run_id: str | None = None,
    auto_launch_full: bool = False,
    force_level: bool = False,
    max_levels: int | None = None,
    run_full_checks: bool = True,
    progress_ui: str = "auto",
    progress_language: str = "zh-CN",
    resume_policy: str = "none",
    resume_stage_checkpoints: Mapping[str, Path] | None = None,
    full_gallery_output_root: Path | None = None,
    full_gallery_samples_per_group: int | None = None,
    reporter: TrainingReporter | None = None,
) -> RampRunResult:
    selected_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / selected_run_id
    _init_layout(output_dir)
    if reporter is not None:
        return _run_training_ramp_with_reporter(
            config_path=config_path,
            device=device,
            target_config_path=target_config_path,
            run_id=selected_run_id,
            output_dir=output_dir,
            auto_launch_full=auto_launch_full,
            force_level=force_level,
            max_levels=max_levels,
            run_full_checks=run_full_checks,
            reporter=reporter,
            resume_policy=resume_policy,
            resume_stage_checkpoints=dict(resume_stage_checkpoints or {}),
            full_gallery_output_root=full_gallery_output_root,
            full_gallery_samples_per_group=full_gallery_samples_per_group,
        )
    dashboard_handle = create_dashboard_reporter(
        run_id=selected_run_id,
        output_dir=output_dir / "dashboard",
        progress_ui=progress_ui,
        progress_language=progress_language,
    )
    with dashboard_handle as dashboard:
        return _run_training_ramp_with_reporter(
            config_path=config_path,
            device=device,
            target_config_path=target_config_path,
            run_id=selected_run_id,
            output_dir=output_dir,
            auto_launch_full=auto_launch_full,
            force_level=force_level,
            max_levels=max_levels,
            run_full_checks=run_full_checks,
            reporter=dashboard.reporter,
            resume_policy=resume_policy,
            resume_stage_checkpoints=dict(resume_stage_checkpoints or {}),
            full_gallery_output_root=full_gallery_output_root,
            full_gallery_samples_per_group=full_gallery_samples_per_group,
        )


def _run_training_ramp_with_reporter(
    *,
    config_path: Path,
    device: str,
    target_config_path: Path | None,
    run_id: str,
    output_dir: Path,
    auto_launch_full: bool,
    force_level: bool,
    max_levels: int | None,
    run_full_checks: bool,
    reporter: TrainingReporter,
    resume_policy: str,
    resume_stage_checkpoints: Mapping[str, Path],
    full_gallery_output_root: Path | None,
    full_gallery_samples_per_group: int | None,
) -> RampRunResult:
    target_config = target_config_path or DEFAULT_FULL_CONFIG
    resolved_target_config, target = ensure_full_target_config(
        source_config=config_path,
        target_config=target_config,
        output_dir=output_dir,
    )
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "git": collect_code_version().as_dict(),
        "source_config": str(config_path),
        "resolved_target_config": str(resolved_target_config),
        "device": device,
        "levels": [],
        "status": "running",
        "full_training": None,
        "resume_policy": resume_policy,
        "resume_stage_checkpoints": {
            stage: str(path) for stage, path in resume_stage_checkpoints.items()
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    reporter.emit_event(
        TrainingEvent.create(
            event_type="ramp",
            severity="info",
            message_key="dashboard_started",
            message_args={"run_id": run_id},
        )
    )

    active_level: RampLevelSpec | None = None
    active_index = 0
    levels = build_ramp_levels(target)
    if max_levels is not None:
        levels = levels[:max_levels]
    try:
        preflight = _run_preflight(
            config_path=config_path,
            device=device,
            output_dir=output_dir,
            run_full_checks=run_full_checks,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        _record_ramp_interrupted(
            manifest=manifest,
            output_dir=output_dir,
            target=target,
            levels=levels,
            auto_launch_full=auto_launch_full,
            reporter=reporter,
            active_index=0,
        )
        raise
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(UTC).isoformat()
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(output_dir / "manifest.json", manifest)
        _report_ramp_failed(
            reporter,
            error=error,
            active_level=None,
            active_index=0,
            completed_levels=0,
            total_levels=0,
        )
        raise
    manifest["preflight"] = preflight
    _write_json(output_dir / "manifest.json", manifest)

    _report_ramp_started(
        reporter,
        levels=levels,
        target=target,
        auto_launch_full=auto_launch_full,
    )
    completed = "level_00_preflight"
    try:
        for index, level in enumerate(levels, start=1):
            active_level = level
            active_index = index
            _report_level_started(
                reporter,
                level=level,
                index=index,
                total_levels=len(levels),
            )
            level_dir = output_dir / f"level_{index:02d}_{level.key}"
            state_path = level_dir / "level_state.json"
            if (
                not force_level
                and state_path.exists()
                and _read_json(state_path).get("status") == "passed"
            ):
                # 渐进训练只在完整通过的 level 边界恢复，避免复用半完成预算或不完整评估。
                record = _read_json(state_path)
                manifest["levels"].append(record)
                _report_level_finished(
                    reporter,
                    level=level,
                    index=index,
                    total_levels=len(levels),
                    record=record,
                    restored=True,
                )
                completed = level.key
                continue
            record = _run_level(
                level=level,
                base_config=config_path,
                level_dir=level_dir,
                device=device,
                reporter=reporter,
                resume_policy=resume_policy,
                resume_stage_checkpoints=resume_stage_checkpoints,
                gallery_output_root=full_gallery_output_root,
                gallery_samples_per_group=full_gallery_samples_per_group,
            )
            manifest["levels"].append(record)
            # 每级结束立即持久化，下一次运行可从最后一个已通过门禁继续。
            _write_json(output_dir / "manifest.json", manifest)
            _report_level_finished(
                reporter,
                level=level,
                index=index,
                total_levels=len(levels),
                record=record,
            )
            completed = level.key
        readiness = _write_final_readiness(
            output_dir=output_dir,
            manifest=manifest,
            target=target,
            levels=levels,
            auto_launch_full=auto_launch_full,
        )
        _report_ramp_finished(
            reporter,
            levels=levels,
            readiness_path=readiness,
            auto_launch_full=auto_launch_full,
        )
        full_run_dir = None
        full_started = False
        if auto_launch_full:
            full_run_dir = output_dir / "full_training"
            _report_full_training_started(reporter, level=target.as_level())
            full_record = _launch_full_training(
                level=target.as_level(),
                config_path=resolved_target_config,
                run_dir=full_run_dir,
                device=device,
                reporter=reporter,
                resume_policy=resume_policy,
                resume_stage_checkpoints=resume_stage_checkpoints,
                gallery_output_root=full_gallery_output_root,
                gallery_samples_per_group=full_gallery_samples_per_group,
            )
            manifest["full_training"] = full_record
            full_started = True
            _report_full_training_finished(reporter, record=full_record)
        manifest["status"] = "passed"
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(output_dir / "manifest.json", manifest)
        return RampRunResult(
            output_dir=output_dir,
            manifest_path=output_dir / "manifest.json",
            final_readiness_path=readiness,
            completed_level=completed,
            full_training_started=full_started,
            full_training_run_dir=full_run_dir,
            status="passed",
        )
    except KeyboardInterrupt:
        _record_ramp_interrupted(
            manifest=manifest,
            output_dir=output_dir,
            target=target,
            levels=levels,
            auto_launch_full=auto_launch_full,
            reporter=reporter,
            active_index=active_index,
        )
        raise
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(UTC).isoformat()
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(output_dir / "manifest.json", manifest)
        _write_final_readiness(
            output_dir=output_dir,
            manifest=manifest,
            target=target,
            levels=levels,
            auto_launch_full=auto_launch_full,
            failure=str(error),
        )
        _report_ramp_failed(
            reporter,
            error=error,
            active_level=active_level,
            active_index=active_index,
            completed_levels=len(
                [
                    item
                    for item in manifest.get("levels", ())
                    if item.get("status") == "passed"
                ]
            ),
            total_levels=len(levels),
        )
        raise


def ensure_full_target_config(
    *,
    source_config: Path,
    target_config: Path,
    output_dir: Path,
) -> tuple[Path, RampTarget]:
    source = _absolutize_config(_read_yaml(source_config), source_config.parent)
    if target_config.exists():
        target_raw = _absolutize_config(_read_yaml(target_config), target_config.parent)
    else:
        target_raw = _build_default_full_config(source)
        target_config.parent.mkdir(parents=True, exist_ok=True)
        _write_yaml(target_config, target_raw)
    target = _target_from_raw(target_raw)
    resolved = output_dir / "resolved_target_config.yaml"
    _write_yaml(resolved, target_raw)
    return resolved, target


def build_ramp_levels(target: RampTarget) -> list[RampLevelSpec]:
    """生成单调递增且每个维度都不超过最终目标的预算序列。"""

    base = [
        RampLevelSpec("a", "level_a", 100, 100, 2, 500, 32, 16, 2),
        RampLevelSpec("b", "level_b", 300, 300, 4, 1500, 64, 16, 4),
    ]
    levels: list[RampLevelSpec] = []
    for level in base:
        clipped = _clip_level(level, target)
        if not levels or clipped.as_dict() != levels[-1].as_dict():
            levels.append(clipped)
        if _level_reaches_target(clipped, target):
            return levels
    current = levels[-1] if levels else base[0]
    index = 3
    while not _level_reaches_target(current, target):
        next_level = RampLevelSpec(
            key=f"c{index - 2}",
            label=f"level_c{index - 2}",
            spatial_steps=min(
                target.spatial_steps,
                max(current.spatial_steps + 1, round(current.spatial_steps * 1.6)),
            ),
            temporal_steps=min(
                target.temporal_steps,
                max(current.temporal_steps + 1, round(current.temporal_steps * 1.6)),
            ),
            patch_limit=min(
                target.patch_limit,
                max(
                    current.patch_limit,
                    current.patch_limit
                    + (1 if current.patch_limit < target.patch_limit else 0),
                ),
            ),
            cache_frames=min(
                target.cache_frames,
                max(current.cache_frames + 1, round(current.cache_frames * 1.6)),
            ),
            sequence_length=min(
                target.sequence_length,
                max(current.sequence_length, round(current.sequence_length * 1.5)),
            ),
            candidate_slots=min(
                target.candidate_slots,
                max(current.candidate_slots, current.candidate_slots + 8),
            ),
            gallery_samples_per_group=min(
                target.gallery_samples_per_group,
                max(
                    current.gallery_samples_per_group,
                    current.gallery_samples_per_group + 1,
                ),
            ),
        )
        levels.append(next_level)
        current = next_level
        index += 1
    return levels


def _run_preflight(
    *,
    config_path: Path,
    device: str,
    output_dir: Path,
    run_full_checks: bool,
    reporter: TrainingReporter,
) -> dict[str, Any]:
    # CUDA、磁盘和数据质量门禁先于任何昂贵训练，失败时不产生误导性的 level 产物。
    preflight_dir = output_dir / "level_00_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    env = collect_environment_report()
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.PRETRAIN_CHECK.value,
        phase="训练预检",
        status="checking",
        trial_status=None,
    )
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="gpu_bridge",
            name="GPU bridge",
            status="checking",
        )
    )
    if device == "cuda" and not env.torch.cuda_available:
        error_message = "CUDA is not visible; run ramp-to-full through host-exec"
        reporter.update_pipeline_stage(
            PipelineStageState(
                stage_id="gpu_bridge",
                name="GPU bridge",
                status="failed",
                processed=0,
                total=1,
                error_reason=error_message,
                blocks_training=True,
                message=error_message,
            )
        )
        raise RampGateError(error_message)
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="gpu_bridge",
            name="GPU bridge",
            status="passed" if env.torch.cuda_available else "skipped",
            processed=1,
            total=1,
            message=(
                f"CUDA 可见：{env.torch.gpu_name or '未知 GPU'}"
                if env.torch.cuda_available
                else "当前运行配置不要求 CUDA"
            ),
        )
    )
    settings = load_settings(config_path)
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="data_check",
            name="数据质量检查",
            status="checking",
        )
    )
    report = inspect_data_input(settings, split="train")
    if not report.ok:
        raise RampGateError("data-check failed")
    free_bytes = _check_output_disk_space(output_dir, reporter=reporter)
    full_checks = None
    if run_full_checks:
        log_path = preflight_dir / "full_checks.log"
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "src/traning/tests/full_checks"],
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        full_checks = {"returncode": completed.returncode, "log": str(log_path)}
        if completed.returncode != 0:
            raise RampGateError(f"full_checks failed; see {log_path}")
    result = {
        "status": "passed",
        "environment": {
            "python": env.python_version,
            "torch": env.torch.version,
            "torch_cuda": env.torch.torch_cuda,
            "cuda_available": env.torch.cuda_available,
            "gpu": env.torch.gpu_name,
            "total_vram_gib": env.torch.total_vram_gib,
            "free_vram_gib": env.torch.free_vram_gib,
        },
        "data": {
            "segments": report.segment_count,
            "estimated_frames": report.frame_count_estimate,
            "categories": report.category_counts,
            "dimensions": report.dimension_counts,
            "slider_quality_issues": tuple(
                item
                for item in report.distribution.get("data_quality_issues", ())
                if "slider" in str(item)
            ),
        },
        "free_disk_gib": free_bytes / 1024**3,
        "full_checks": full_checks,
    }
    reporter.report_resource(collect_resource_state())
    reporter.report_dataset_usage(
        DatasetUsageState(
            total_segments=report.segment_count,
            total_frames=report.frame_count_estimate,
        )
    )
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="data_check",
            name="数据质量检查",
            status="warning" if result["data"]["slider_quality_issues"] else "passed",
            processed=report.segment_count,
            total=report.segment_count,
            warning_count=len(result["data"]["slider_quality_issues"]),
            blocks_training=False,
        )
    )
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="output_disk",
            name="输出空间检查",
            status="passed",
            processed=1,
            total=1,
            message=f"可用空间 {free_bytes / 1024**3:.2f} GiB",
        )
    )
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="readiness",
            name="训练 readiness",
            status="passed",
            processed=1,
            total=1,
            output_path=str(preflight_dir / "preflight.json"),
        )
    )
    _write_json(preflight_dir / "preflight.json", result)
    return result


def _free_disk_bytes(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def _minimum_ramp_output_free_bytes() -> int:
    raw = os.environ.get("OSU_AI_MIN_RAMP_OUTPUT_FREE_GIB")
    minimum_gib = 10.0 if raw is None else float(raw)
    return int(minimum_gib * 1024**3)


def _check_output_disk_space(
    output_dir: Path,
    *,
    reporter: TrainingReporter,
) -> int:
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="output_disk",
            name="输出空间检查",
            status="checking",
        )
    )
    free_bytes = _free_disk_bytes(output_dir)
    minimum_bytes = _minimum_ramp_output_free_bytes()
    if free_bytes < minimum_bytes:
        message = (
            f"less than {minimum_bytes / 1024**3:.1f} GiB free space for ramp outputs; "
            f"available {free_bytes / 1024**3:.2f} GiB"
        )
        reporter.update_pipeline_stage(
            PipelineStageState(
                stage_id="output_disk",
                name="输出空间检查",
                status="failed",
                processed=0,
                total=1,
                blocks_training=True,
                error_reason=message,
                message=message,
            )
        )
        raise RampGateError(message)
    return free_bytes


def _report_ramp_started(
    reporter: TrainingReporter,
    *,
    levels: list[RampLevelSpec],
    target: RampTarget,
    auto_launch_full: bool,
) -> None:
    total_steps = sum(level.spatial_steps + level.temporal_steps for level in levels)
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="training_ramp",
            name="受控渐进放大",
            status="running",
            processed=0,
            total=len(levels),
            message="UI 已启动，准备自动执行渐进训练",
        )
    )
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.PROGRESSIVE_PREPARATION.value,
        phase="受控渐进放大准备",
        status="running",
        current_level=None,
        trial_status=None,
        completed_levels=0,
        total_levels=len(levels),
        global_step=0,
        target_global_steps=total_steps,
        spatial_step=0,
        spatial_target=target.spatial_steps,
        temporal_step=0,
        temporal_target=target.temporal_steps,
        current_grade="observing",
        best_grade=None,
        promotion_status=(
            "渐进训练启动，完成后自动进入正式训练"
            if auto_launch_full
            else "渐进训练启动"
        ),
        consecutive_passes=0,
        required_passes=len(levels),
    )
    reporter.emit_event(
        TrainingEvent.create(
            event_type="ramp",
            severity="info",
            message_key="stage_started",
            message_args={"stage": "受控渐进放大"},
        )
    )


def _report_level_started(
    reporter: TrainingReporter,
    *,
    level: RampLevelSpec,
    index: int,
    total_levels: int,
) -> None:
    stage_id = _level_stage_id(level)
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id=stage_id,
            name=_level_title(level),
            status="running",
            processed=0,
            total=level.spatial_steps + level.temporal_steps,
            message="正在训练并等待 gate 判定",
        )
    )
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.TRAINING.value,
        phase=_level_title(level),
        status="running",
        current_level=level.key,
        trial_status="training",
        completed_levels=index - 1,
        total_levels=total_levels,
        global_step=0,
        target_global_steps=level.spatial_steps + level.temporal_steps,
        spatial_step=0,
        spatial_target=level.spatial_steps,
        temporal_step=0,
        temporal_target=level.temporal_steps,
        current_trial_id=f"ramp-{level.key}",
        current_grade="observing",
        promotion_status=f"{_level_title(level)} 正在训练",
        consecutive_passes=index - 1,
        required_passes=total_levels,
        current_parameters={"ramp_level": level.as_dict()},
    )
    reporter.emit_event(
        TrainingEvent.create(
            event_type="ramp_level",
            severity="info",
            message_key="stage_started",
            message_args={"stage": _level_title(level)},
            level=level.key,
            trial_id=f"ramp-{level.key}",
        )
    )


def _report_level_finished(
    reporter: TrainingReporter,
    *,
    level: RampLevelSpec,
    index: int,
    total_levels: int,
    record: Mapping[str, Any],
    restored: bool = False,
) -> None:
    score = _record_quality_score(record)
    threshold = _record_pass_threshold(record)
    gallery_path = _record_gallery_path(record)
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id=_level_stage_id(level),
            name=_level_title(level),
            status="passed",
            processed=level.spatial_steps + level.temporal_steps,
            total=level.spatial_steps + level.temporal_steps,
            output_path=str(record.get("artifact_manifest") or gallery_path or ""),
            message="已通过 gate" + ("，来自已通过记录" if restored else ""),
            score=score,
            threshold=threshold,
        )
    )
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="training_ramp",
            name="受控渐进放大",
            status="running" if index < total_levels else "passed",
            processed=index,
            total=total_levels,
            output_path=str(record.get("artifact_manifest") or ""),
            message=f"{_level_title(level)} 已通过",
        )
    )
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.TRAINING.value,
        phase=_level_title(level),
        status="running" if index < total_levels else "passed",
        current_level=level.key,
        trial_status="promoted" if index < total_levels else "passed",
        completed_levels=index,
        total_levels=total_levels,
        global_step=level.spatial_steps + level.temporal_steps,
        target_global_steps=level.spatial_steps + level.temporal_steps,
        spatial_step=level.spatial_steps,
        spatial_target=level.spatial_steps,
        temporal_step=level.temporal_steps,
        temporal_target=level.temporal_steps,
        score=score,
        parameter_best_score=score,
        level_best_score=score,
        frames_per_second=record.get("frames_per_second"),
        steps_per_second=record.get("steps_per_second"),
        current_grade="reached",
        best_grade="reached",
        promotion_status=f"{_level_title(level)} 已通过 gate",
        consecutive_passes=index,
        required_passes=total_levels,
    )
    if restored and score is not None:
        reporter.report_score(
            score=score, trial_id=f"ramp-{level.key}", level=level.key
        )
    if gallery_path:
        reporter.emit_event(
            TrainingEvent.create(
                event_type="gallery",
                severity="success",
                message_key="gallery_saved",
                message_args={"path": gallery_path},
                level=level.key,
                trial_id=f"ramp-{level.key}",
            )
        )
    reporter.emit_event(
        TrainingEvent.create(
            event_type="ramp_level",
            severity="success",
            message_key="stage_finished",
            message_args={"stage": _level_title(level), "status": "已通过"},
            level=level.key,
            trial_id=f"ramp-{level.key}",
        )
    )


def _report_ramp_finished(
    reporter: TrainingReporter,
    *,
    levels: list[RampLevelSpec],
    readiness_path: Path,
    auto_launch_full: bool,
) -> None:
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="final_readiness",
            name="最终 readiness",
            status="passed",
            processed=1,
            total=1,
            output_path=str(readiness_path),
            message="渐进训练已全部通过",
        )
    )
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.TRAINING.value,
        phase="最终 readiness",
        status="passed" if not auto_launch_full else "running",
        trial_status="promoted" if auto_launch_full else "passed",
        current_grade="promotable",
        best_grade="promotable",
        promotion_status=(
            "全部 Level 已通过，正在进入正式训练"
            if auto_launch_full
            else "全部 Level 已通过"
        ),
        completed_levels=len(levels),
        total_levels=len(levels),
        consecutive_passes=len(levels),
        required_passes=len(levels),
    )
    reporter.emit_event(
        TrainingEvent.create(
            event_type="ramp",
            severity="success",
            message_key="stage_finished",
            message_args={"stage": "受控渐进放大", "status": "已通过"},
        )
    )


def _report_full_training_started(
    reporter: TrainingReporter,
    *,
    level: RampLevelSpec,
) -> None:
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="full_training",
            name="全模型正式训练",
            status="running",
            processed=0,
            total=level.spatial_steps + level.temporal_steps,
            message="渐进训练通过后自动启动",
        )
    )
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.TRAINING.value,
        phase="全模型正式训练",
        status="running",
        current_level=level.key,
        current_trial_id="full-model",
        trial_status="training",
        global_step=0,
        target_global_steps=level.spatial_steps + level.temporal_steps,
        spatial_step=0,
        spatial_target=level.spatial_steps,
        temporal_step=0,
        temporal_target=level.temporal_steps,
        current_grade="promoted",
        promotion_status="正式训练进行中",
    )


def _report_full_training_finished(
    reporter: TrainingReporter,
    *,
    record: Mapping[str, Any],
) -> None:
    summary = dict(record.get("summary") or {})
    score = _summary_quality_score(summary)
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="full_training",
            name="全模型正式训练",
            status="passed",
            processed=int(summary.get("spatial_steps", 0))
            + int(summary.get("temporal_steps", 0)),
            total=None,
            output_path=str(record.get("run_dir") or ""),
            message="正式训练完成",
        )
    )
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.TRAINING.value,
        phase="全模型正式训练",
        status="passed",
        score=score,
        trial_status="completed",
        current_grade="promoted",
        best_grade="promoted",
        promotion_status="正式训练完成",
    )
    if score is not None:
        reporter.report_score(score=score, trial_id="full-model", level="target")


def _report_ramp_failed(
    reporter: TrainingReporter,
    *,
    error: Exception,
    active_level: RampLevelSpec | None,
    active_index: int,
    completed_levels: int,
    total_levels: int,
) -> None:
    message = f"{type(error).__name__}: {error}"
    if active_level is not None:
        reporter.update_pipeline_stage(
            PipelineStageState(
                stage_id=_level_stage_id(active_level),
                name=_level_title(active_level),
                status="failed",
                processed=0,
                total=active_level.spatial_steps + active_level.temporal_steps,
                error_reason=message,
                blocks_training=True,
                message="未通过 gate 或训练异常",
            )
        )
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="training_ramp",
            name="受控渐进放大",
            status="failed",
            processed=completed_levels,
            total=total_levels,
            error_reason=message,
            blocks_training=True,
            message="渐进训练失败",
        )
    )
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.FAILED.value,
        phase=_level_title(active_level)
        if active_level is not None
        else "受控渐进放大",
        status="failed",
        current_level=active_level.key if active_level is not None else None,
        trial_status="failed",
        completed_levels=completed_levels,
        total_levels=total_levels,
        current_grade="stopped",
        promotion_status=(
            f"{_level_title(active_level)} 失败"
            if active_level is not None
            else "渐进训练失败"
        ),
        consecutive_passes=completed_levels,
        required_passes=total_levels,
    )
    reporter.request_stop(
        TrainingStopState(
            reason="RAMP_FAILED",
            message=message,
            exit_code=1,
            step=active_index or None,
            target_step=total_levels or None,
        )
    )


def _record_ramp_interrupted(
    *,
    manifest: dict[str, Any],
    output_dir: Path,
    target: RampTarget,
    levels: list[RampLevelSpec],
    auto_launch_full: bool,
    reporter: TrainingReporter,
    active_index: int,
) -> None:
    """在用户中断时落盘可恢复状态，而不是把 manifest 留在 running。"""

    interrupted_at = datetime.now(UTC).isoformat()
    manifest["status"] = "interrupted"
    manifest["interrupted_at_utc"] = interrupted_at
    manifest["failure"] = {
        "type": "KeyboardInterrupt",
        "message": "用户中断训练参数搜索",
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_final_readiness(
        output_dir=output_dir,
        manifest=manifest,
        target=target,
        levels=levels,
        auto_launch_full=auto_launch_full,
        failure="用户中断训练参数搜索",
        status="interrupted",
        updated_at_utc=interrupted_at,
    )
    reporter.request_stop(
        TrainingStopState(
            reason="USER_INTERRUPTED",
            message="用户中断训练参数搜索",
            exit_code=2,
            step=active_index or None,
            target_step=len(levels) or None,
        )
    )


def _level_stage_id(level: RampLevelSpec) -> str:
    return f"level_{level.key}"


def _level_title(level: RampLevelSpec | None) -> str:
    if level is None:
        return "受控渐进放大"
    return f"Level {level.key.upper()}"


def _record_quality_score(record: Mapping[str, Any]) -> float | None:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return None
    value = evaluation.get("quality_score")
    return float(value) if value is not None else None


def _record_pass_threshold(record: Mapping[str, Any]) -> float | None:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return None
    value = evaluation.get("pass_threshold")
    return float(value) if value is not None else None


def _summary_quality_score(summary: Mapping[str, Any]) -> float | None:
    value = summary.get("quality_score")
    return float(value) if value is not None else None


def _record_gallery_path(record: Mapping[str, Any]) -> str | None:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return None
    value = evaluation.get("gallery_output_dir")
    return str(value) if value else None


def _run_level(
    *,
    level: RampLevelSpec,
    base_config: Path,
    level_dir: Path,
    device: str,
    reporter: TrainingReporter,
    resume_policy: str,
    resume_stage_checkpoints: Mapping[str, Path],
    gallery_output_root: Path | None,
    gallery_samples_per_group: int | None,
) -> dict[str, Any]:
    level_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_level_config(base_config, level_dir, level)
    started = time.monotonic()
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id=f"level_{level.key}",
            name=f"Level {level.key.upper()}",
            status="running",
            total=level.spatial_steps + level.temporal_steps,
        )
    )
    settings = load_settings(config_path)
    optimization = getattr(settings, "optimization", None)
    max_trials = getattr(optimization, "max_trials", 1)
    execute_generated_jobs = bool(
        getattr(optimization, "execute_generated_jobs", False)
    )
    resumed = _load_pending_search(level_dir)
    trial_job: Mapping[str, Any] | None = None
    trial_failures: list[str] = []
    trial_index = 0
    if resumed is not None:
        state, pending_path = resumed
        attempted_trials = int(state["attempted_trials"])
        exhausted = max_trials is not None and attempted_trials >= max_trials
        if not execute_generated_jobs or exhausted:
            budget_text = "unbounded" if max_trials is None else str(max_trials)
            reason = (
                "automatic generated-job execution is disabled"
                if not execute_generated_jobs
                else f"trial budget exhausted ({attempted_trials}/{budget_text})"
            )
            raise RampSearchExhausted(
                f"{reason}; pending next job: {pending_path}; "
                f"last evaluation: {state.get('last_error') or 'unknown'}"
            )
        pending_job = _load_next_job(pending_path)
        if pending_job is None:
            raise RampGateError(f"pending next job is unavailable: {pending_path}")
        trial_job = pending_job
        trial_index = attempted_trials
        if state.get("last_error"):
            trial_failures.append(str(state["last_error"]))
        reporter.emit_event(
            TrainingEvent.create(
                event_type="ramp.resume",
                severity="info",
                message_key="stage_started",
                message_args={"stage": _level_title(level)},
                raw_message=(
                    "[RAMP][RESUME_SEARCH] "
                    f"level={level.key} attempted={attempted_trials} "
                    f"next_trial={pending_job.get('trial_id')}"
                ),
            )
        )
    record: dict[str, Any] | None = None
    while True:
        try:
            record = _run_level_trial(
                level=level,
                settings=settings,
                config_path=config_path,
                level_dir=level_dir,
                device=device,
                reporter=reporter,
                resume_policy=resume_policy,
                resume_stage_checkpoints=resume_stage_checkpoints,
                gallery_output_root=gallery_output_root,
                gallery_samples_per_group=gallery_samples_per_group,
                started=started,
                trial_index=trial_index,
                trial_job=trial_job,
            )
            break
        except RampEvaluationGateError as error:
            trial_failures.append(str(error))
            next_job_path = _trial_next_job_path(level_dir, trial_index)
            next_job = _load_next_job(next_job_path)
            attempted_trials = trial_index + 1
            if next_job is None:
                raise RampGateError(
                    f"evaluation did not pass and next job is unavailable: {next_job_path}"
                ) from error
            exhausted = max_trials is not None and attempted_trials >= max_trials
            search_status = (
                "execution_disabled"
                if not execute_generated_jobs
                else "search_exhausted"
                if exhausted
                else "pending"
            )
            _write_json(
                level_dir / "search_state.json",
                {
                    "status": search_status,
                    "level": level.key,
                    "attempted_trials": attempted_trials,
                    "max_trials": max_trials,
                    "execute_generated_jobs": execute_generated_jobs,
                    "last_error": str(error),
                    "pending_next_job": str(next_job_path.resolve()),
                    "pending_trial_id": next_job.get("trial_id"),
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                },
            )
            if not execute_generated_jobs or exhausted:
                budget_text = "unbounded" if max_trials is None else str(max_trials)
                reason = (
                    "automatic generated-job execution is disabled"
                    if not execute_generated_jobs
                    else f"trial budget exhausted ({attempted_trials}/{budget_text})"
                )
                reporter.emit_event(
                    TrainingEvent.create(
                        event_type="ramp.search_exhausted",
                        severity="error",
                        message_key="fatal_error",
                        message_args={"error": reason},
                        raw_message=(
                            "[RAMP][SEARCH_EXHAUSTED] "
                            f"level={level.key} trials={attempted_trials}/{budget_text} "
                            f"pending_job={next_job_path} reason={reason}"
                        ),
                    )
                )
                raise RampSearchExhausted(
                    f"{reason}; pending next job: {next_job_path}; "
                    f"last evaluation: {error}"
                ) from error
            reporter.emit_event(
                TrainingEvent.create(
                    event_type="ramp.continue",
                    severity="warning",
                    message_key="stage_warning",
                    message_args={"stage": _level_title(level)},
                    raw_message=(
                        "[RAMP][CONTINUE] "
                        f"level={level.key} trial={attempted_trials}/"
                        f"{max_trials if max_trials is not None else 'unbounded'} "
                        f"next_trial={next_job.get('trial_id')} reason={error}"
                    ),
                )
            )
            trial_job = next_job
            trial_index += 1
    if record is None:
        raise RampGateError(
            "; ".join(trial_failures) or "level produced no trial record"
        )
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.TRAINING.value,
        phase=_level_title(level),
        status="passed",
        current_level=level.key,
        trial_status="passed",
        current_parameters=_ramp_parameter_snapshot(
            level=level,
            record=record,
            config_path=config_path,
            device=device,
            resume_policy=resume_policy,
            resume_stage_checkpoints=resume_stage_checkpoints,
        ),
    )
    reporter.report_score(
        score=float(record["evaluation"]["quality_score"]),
        trial_id=f"ramp-{level.key}",
        level=level.key,
        category_scores={"slider": record["slider_score"]}
        if record["slider_score"] is not None
        else None,
    )
    reporter.report_resource(collect_resource_state())
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id=f"level_{level.key}",
            name=f"Level {level.key.upper()}",
            status="passed",
            processed=level.spatial_steps + level.temporal_steps,
            total=level.spatial_steps + level.temporal_steps,
            output_path=str(level_dir / "level_state.json"),
            score=float(record["evaluation"]["quality_score"]),
            threshold=record["evaluation"].get("pass_threshold"),
        )
    )
    _write_json(level_dir / "level_state.json", record)
    _write_json(
        level_dir / "search_state.json",
        {
            "status": "passed",
            "level": level.key,
            "attempted_trials": trial_index + 1,
            "max_trials": max_trials,
            "execute_generated_jobs": execute_generated_jobs,
            "selected_trial_id": record["evaluation"].get("parameter_group_id"),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return record


def _run_level_trial(
    *,
    level: RampLevelSpec,
    settings,
    config_path: Path,
    level_dir: Path,
    device: str,
    reporter: TrainingReporter,
    resume_policy: str,
    resume_stage_checkpoints: Mapping[str, Path],
    gallery_output_root: Path | None,
    gallery_samples_per_group: int | None,
    started: float,
    trial_index: int,
    trial_job: Mapping[str, Any] | None,
) -> dict[str, Any]:
    trial_id = str((trial_job or {}).get("trial_id") or f"ramp-{level.key}")
    optimization_stage = _trial_curriculum_stage(trial_job)
    optimization_rung = _trial_rung(trial_job)
    budget_steps = int((trial_job or {}).get("budget_steps") or level.temporal_steps)
    run_dir = level_dir / (
        "training" if trial_index == 0 else f"training_trial_{trial_index + 1:02d}"
    )
    stage_checkpoints = dict(resume_stage_checkpoints)
    parent_checkpoint = (trial_job or {}).get("parent_checkpoint_path")
    if isinstance(parent_checkpoint, str) and parent_checkpoint:
        stage_checkpoints["temporal"] = Path(parent_checkpoint)
    # 新 trial 的学习率或候选槽可能变化；只继承模型权重，避免旧 optimizer
    # 覆盖本轮参数，并把 budget_steps 解释为本轮独立预算。
    trial_resume_policy = "weights-only" if parent_checkpoint else resume_policy
    runtime = _trial_runtime_overrides(
        settings=settings,
        level=level,
        trial_index=trial_index,
        budget_steps=budget_steps,
        trial_job=trial_job,
        parent_checkpoint_path=stage_checkpoints.get("temporal"),
    )
    reporter.emit_event(
        TrainingEvent.create(
            event_type="optimizer.select",
            severity="info",
            message_key="stage_started",
            message_args={"stage": "optimizer trial"},
            raw_message=(
                "[OPTIMIZER][SELECT] "
                f"trial={trial_id} budget_steps={budget_steps} "
                f"spatial_steps={runtime['spatial_max_steps']} "
                f"temporal_steps={runtime['temporal_max_steps']} "
                f"score_threshold={runtime['score_threshold']} "
                f"max_candidates={runtime['max_candidates']} "
                f"parent={parent_checkpoint or 'none'}"
            ),
        )
    )
    result = run_full_training_pipeline(
        settings,
        config=FullTrainingRunConfig(
            run_dir=run_dir,
            device=torch.device(device),
            spatial_max_steps=int(runtime["spatial_max_steps"]),
            temporal_max_steps=int(runtime["temporal_max_steps"]),
            spatial_learning_rate=float(runtime["spatial_learning_rate"]),
            temporal_learning_rate=float(runtime["temporal_learning_rate"]),
            patch_limit=runtime["patch_limit"],
            cache_max_frames=runtime["cache_max_frames"],
            max_candidates=runtime["max_candidates"],
            score_threshold=runtime["score_threshold"],
            nms_radius_px=runtime["nms_radius_px"],
            slider_threshold=runtime["slider_threshold"],
            max_slider_paths=runtime["max_slider_paths"],
            sequence_length=int(runtime["sequence_length"]),
            candidate_slots=int(runtime["candidate_slots"]),
            parameter_group_id=trial_id,
            curriculum_level=level.key,
            optimization_stage=optimization_stage,
            optimization_rung=optimization_rung,
            render_gallery=True,
            gallery_output_root=gallery_output_root,
            gallery_samples_per_group=(
                gallery_samples_per_group or level.gallery_samples_per_group
            ),
            reporter=reporter,
            resume_policy=trial_resume_policy,
            resume_stage_checkpoints=stage_checkpoints,
        ),
    )
    elapsed = time.monotonic() - started
    artifact = export_model_artifact(
        ModelArtifactSpec(
            artifact_id=f"artifact-{level.key}-{trial_index + 1:02d}",
            output_dir=level_dir / "artifacts",
            settings_path=config_path,
            spatial_checkpoint_path=result.spatial.checkpoint_path,
            temporal_checkpoint_path=result.temporal.checkpoint_path,
            score_version=AGGREGATE_SCORE_VERSION,
            candidate_cache_version=CANDIDATE_CACHE_VERSION,
            code_version=collect_code_version().commit,
            extra_files={
                "score_report": result.evaluation.report_path,
                "gallery_request": result.evaluation.gallery_request_path,
                "summary": result.summary_path,
                "candidate_cache_manifest": result.candidate_cache.manifest_path,
            },
        )
    )
    artifact_issues = validate_model_artifact(artifact.manifest_path)
    artifact_smoke = smoke_test_model_artifact(artifact.manifest_path, device="cpu")
    dry_run = _run_job_dry_run(
        job_path=result.evaluation.next_job_path,
        config_path=config_path,
        level_dir=level_dir,
        device=device,
    )
    return _gate_level(
        level=level,
        result=result,
        expected_spatial_steps=int(runtime["spatial_max_steps"]),
        expected_temporal_steps=int(runtime["temporal_max_steps"]),
        runtime=runtime,
        elapsed=elapsed,
        artifact_path=artifact.manifest_path,
        artifact_issues=artifact_issues,
        artifact_smoke=artifact_smoke,
        dry_run=dry_run,
    )


def _trial_next_job_path(level_dir: Path, trial_index: int) -> Path:
    run_dir = level_dir / (
        "training" if trial_index == 0 else f"training_trial_{trial_index + 1:02d}"
    )
    return run_dir / "evaluation" / "next_training_job.json"


def _load_pending_search(
    level_dir: Path,
) -> tuple[dict[str, Any], Path] | None:
    """读取可恢复的搜索边界，避免同一 run_id 重跑初始 trial。"""

    state_path = level_dir / "search_state.json"
    if not state_path.is_file():
        return None
    state = _read_json(state_path)
    if state.get("status") not in {
        "pending",
        "search_exhausted",
        "execution_disabled",
    }:
        return None
    try:
        attempted_trials = int(state.get("attempted_trials") or 0)
    except (TypeError, ValueError) as error:
        raise RampGateError(f"invalid search state: {state_path}") from error
    pending_raw = state.get("pending_next_job")
    if attempted_trials < 1 or not pending_raw:
        raise RampGateError(f"incomplete pending search state: {state_path}")
    state["attempted_trials"] = attempted_trials
    return state, Path(str(pending_raw))


def _load_next_job(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    raw = _read_json(path)
    try:
        return training_job_from_dict(raw).as_dict()
    except ValueError as error:
        raise RampGateError(f"invalid next training job {path}: {error}") from error


def _trial_curriculum_stage(
    trial_job: Mapping[str, Any] | None,
) -> CurriculumStage:
    raw = (trial_job or {}).get("curriculum_stage", CurriculumStage.BASIC.value)
    try:
        return CurriculumStage(str(raw))
    except ValueError as error:
        raise RampGateError(f"invalid job curriculum_stage: {raw!r}") from error


def _trial_rung(trial_job: Mapping[str, Any] | None) -> int:
    raw = (trial_job or {}).get("rung", 0)
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise RampGateError(f"invalid job rung: {raw!r}") from error
    if value < 0:
        raise RampGateError("job rung must be nonnegative")
    return value


def _trial_runtime_overrides(
    *,
    settings,
    level: RampLevelSpec,
    trial_index: int,
    budget_steps: int,
    trial_job: Mapping[str, Any] | None,
    parent_checkpoint_path: Path | None,
) -> dict[str, Any]:
    parameters = _mapping((trial_job or {}).get("parameters"))
    training = _mapping(parameters.get("training"))
    inference = _mapping(parameters.get("inference"))
    parent_temporal_step = (
        _checkpoint_temporal_step(parent_checkpoint_path)
        if trial_index > 0 and parent_checkpoint_path is not None
        else 0
    )
    sequence_length = _scaled_positive_int(
        level.sequence_length,
        explicit=training.get("sequence_length"),
        multiplier=training.get("sequence_length_multiplier"),
    )
    candidate_slots = _scaled_positive_int(
        level.candidate_slots,
        explicit=training.get("candidate_slots"),
        multiplier=training.get("candidate_slots_multiplier"),
    )
    spatial_steps = level.spatial_steps if trial_index == 0 else budget_steps
    temporal_budget = level.temporal_steps if trial_index == 0 else budget_steps
    return {
        "spatial_max_steps": spatial_steps,
        "temporal_max_steps": max(1, temporal_budget),
        "temporal_budget_steps": temporal_budget,
        "parent_temporal_step": parent_temporal_step,
        "sequence_length": sequence_length,
        "candidate_slots": candidate_slots,
        "spatial_learning_rate": _positive_float_override(
            1e-4,
            training.get("spatial_learning_rate"),
        ),
        "temporal_learning_rate": _positive_float_override(
            1e-4,
            training.get("temporal_learning_rate"),
        ),
        "patch_limit": _optional_limit_override(
            level.patch_limit,
            training.get("patch_limit"),
        ),
        "cache_max_frames": _optional_limit_override(
            level.cache_frames,
            training.get("cache_max_frames"),
        ),
        "score_threshold": _score_threshold_override(
            _candidate_cache_default(settings, "score_threshold", 0.05),
            inference.get("score_threshold"),
            inference.get("score_threshold_delta"),
        ),
        "max_candidates": _max_candidates_override(
            _candidate_cache_default(settings, "max_candidates_per_frame", 32),
            inference.get("max_candidates"),
            inference.get("max_candidates_delta"),
        ),
        "nms_radius_px": _float_override(
            _candidate_cache_default(settings, "nms_radius_px", 24.0),
            inference.get("nms_radius_px"),
        ),
        "slider_threshold": _float_override(
            _candidate_cache_default(settings, "slider_threshold", 0.35),
            inference.get("slider_threshold"),
        ),
        "max_slider_paths": _positive_int_override(
            _candidate_cache_default(settings, "max_slider_paths", 16),
            inference.get("max_slider_paths"),
        ),
        "parameters": parameters,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _candidate_cache_default(settings: Any, field_name: str, fallback: Any) -> Any:
    candidate_cache = getattr(settings, "candidate_cache", None)
    return getattr(candidate_cache, field_name, fallback)


def _score_threshold_override(
    default: float,
    explicit: Any,
    delta: Any,
) -> float:
    if isinstance(delta, (int, float)):
        return _clamp(float(default) + float(delta), 0.0, 1.0)
    if isinstance(explicit, (int, float)):
        # TrainingJobSpec 保存的是已解析绝对值；旧 job 中越界值也只做边界裁剪，
        # 不再根据正负号猜测它是不是 delta。
        return _clamp(float(explicit), 0.0, 1.0)
    return float(default)


def _max_candidates_override(
    default: int,
    explicit: Any,
    delta: Any,
) -> int:
    if isinstance(delta, (int, float)):
        return max(1, int(round(float(default) + float(delta))))
    if isinstance(explicit, (int, float)):
        return max(1, int(round(float(explicit))))
    return max(1, int(default))


def _scaled_positive_int(
    default: int,
    *,
    explicit: Any,
    multiplier: Any,
) -> int:
    if isinstance(explicit, (int, float)):
        return max(1, int(round(float(explicit))))
    if isinstance(multiplier, (int, float)):
        return max(1, int(round(float(default) * float(multiplier))))
    return max(1, int(default))


def _float_override(default: float, explicit: Any) -> float:
    return float(explicit) if isinstance(explicit, (int, float)) else float(default)


def _positive_float_override(default: float, explicit: Any) -> float:
    if isinstance(explicit, (int, float)):
        return max(1e-7, float(explicit))
    return max(1e-7, float(default))


def _positive_int_override(default: int, explicit: Any) -> int:
    if isinstance(explicit, (int, float)):
        return max(1, int(round(float(explicit))))
    return max(1, int(default))


def _optional_limit_override(default: int, explicit: Any) -> int | None:
    """解析 patch/帧上限；0 与训练 CLI 一样表示不限制。"""

    if isinstance(explicit, (int, float)):
        value = max(0, int(round(float(explicit))))
        return None if value == 0 else value
    return max(1, int(default))


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _checkpoint_temporal_step(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        return 0
    position = TrainingPosition.from_mapping(payload.get("training_position"))
    return max(
        position.temporal_step, position.global_step, position.last_committed_step
    )


def _gate_level(
    *,
    level: RampLevelSpec,
    result,
    elapsed: float,
    artifact_path: Path,
    artifact_issues: tuple[str, ...],
    artifact_smoke: dict[str, Any],
    dry_run: dict[str, Any],
    expected_spatial_steps: int | None = None,
    expected_temporal_steps: int | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    evaluation_failure: str | None = None
    expected_spatial_steps = expected_spatial_steps or level.spatial_steps
    expected_temporal_steps = expected_temporal_steps or level.temporal_steps
    if result.spatial.steps != expected_spatial_steps:
        failures.append("spatial steps did not reach requested level")
    if result.temporal.steps != expected_temporal_steps:
        failures.append("temporal steps did not reach requested level")
    for label, value in {
        "spatial_loss": result.spatial.last_loss,
        "temporal_loss": result.temporal.final_loss,
        "quality_score": result.evaluation.quality_score,
    }.items():
        if not math.isfinite(float(value)):
            failures.append(f"{label} is not finite")
    quality_score = float(result.evaluation.quality_score)
    pass_threshold = float(result.evaluation.pass_threshold)
    if quality_score < pass_threshold:
        evaluation_failure = (
            "quality score "
            f"{quality_score:.6f} below pass threshold {pass_threshold:.6f}"
        )
    elif not result.evaluation.passed:
        target_count = getattr(result.evaluation, "target_count", None)
        if target_count is not None and int(target_count) <= 0:
            evaluation_failure = (
                "evaluation report did not pass: no target frames were evaluated; "
                "background-only accuracy cannot satisfy the promotion gate"
            )
        else:
            evaluation_failure = (
                "evaluation report did not pass despite quality score "
                f"{quality_score:.6f} >= pass threshold {pass_threshold:.6f}; "
                f"hits={result.evaluation.hit_count} "
                f"misses={result.evaluation.miss_count} "
                f"unresolved={result.evaluation.unresolved_count}"
            )
    checkpoint_paths = (
        result.spatial.checkpoint_path,
        result.temporal.checkpoint_path,
    )
    for path in checkpoint_paths:
        if not path.exists():
            failures.append(f"checkpoint missing: {path}")
        else:
            torch.load(path, map_location="cpu", weights_only=False)
    if (
        result.evaluation.gallery_status != "saved"
        or result.evaluation.gallery_saved_frame_count <= 0
    ):
        failures.append("gallery is empty or not saved")
    if not result.evaluation.report_path.exists():
        failures.append("score report missing")
    if (
        result.evaluation.next_job_path is None
        or not result.evaluation.next_job_path.exists()
    ):
        failures.append("next job missing")
    if dry_run.get("returncode") != 0:
        failures.append("run-job --dry-run failed")
    if artifact_issues:
        failures.append("artifact validation failed: " + "; ".join(artifact_issues))
    if not artifact_smoke.get("finite"):
        failures.append("artifact smoke produced non-finite outputs")
    if failures:
        if evaluation_failure is not None:
            failures.insert(0, evaluation_failure)
        raise RampGateError("; ".join(failures))
    if evaluation_failure is not None:
        raise RampEvaluationGateError(evaluation_failure)
    steps_per_second = (level.spatial_steps + level.temporal_steps) / max(elapsed, 1e-6)
    frames_per_second = result.candidate_cache.frames / max(elapsed, 1e-6)
    report = _read_json(result.evaluation.report_path)
    slider_samples = [
        sample
        for sample in report.get("samples", ())
        if "slider" in str(sample.get("subproject", ""))
    ]
    slider_score = (
        sum(float(sample.get("quality_score", 0.0)) for sample in slider_samples)
        / len(slider_samples)
        if slider_samples
        else None
    )
    return {
        "status": "passed",
        "level": level.as_dict(),
        "runtime": _json_ready(dict(runtime or {})),
        "elapsed_seconds": elapsed,
        "steps_per_second": steps_per_second,
        "frames_per_second": frames_per_second,
        "spatial": result.spatial.as_dict(),
        "temporal": result.temporal.as_dict(),
        "candidate_cache": result.candidate_cache.as_dict(),
        "decision": result.decision.as_dict(),
        "evaluation": result.evaluation.as_dict(),
        "artifact_manifest": str(artifact_path),
        "artifact_smoke": artifact_smoke,
        "dry_run": dry_run,
        "slider_score": slider_score,
        "slider_sample_count": len(slider_samples),
        "peak_vram_gib": max(
            value or 0.0
            for value in (
                result.spatial.cuda_max_reserved_gib,
                result.temporal.cuda_max_reserved_gib,
            )
        ),
    }


def _ramp_parameter_snapshot(
    *,
    level: RampLevelSpec,
    record: Mapping[str, Any],
    config_path: Path,
    device: str,
    resume_policy: str,
    resume_stage_checkpoints: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "parameter_group_id": f"ramp-{level.key}",
        "device": device,
        "source_config": str(config_path),
        "ramp": {
            "level": level.as_dict(),
            "steps_per_second": record["steps_per_second"],
            "frames_per_second": record["frames_per_second"],
            "peak_vram_gib": record["peak_vram_gib"],
            "slider_score": record["slider_score"],
            "slider_sample_count": record["slider_sample_count"],
            "resume_policy": resume_policy,
            "resume_stage_checkpoints": {
                stage: str(path) for stage, path in resume_stage_checkpoints.items()
            },
        },
        "evaluation": {
            "quality_score": record["evaluation"]["quality_score"],
            "pass_threshold": record["evaluation"].get("pass_threshold"),
            "passed": record["evaluation"]["passed"],
            "gallery_status": record["evaluation"]["gallery_status"],
            "report_path": record["evaluation"]["report_path"],
            "asha_action": record["evaluation"].get("asha_action"),
            "asha_reasons": record["evaluation"].get("asha_reasons", ()),
        },
        "artifact": {
            "manifest": record["artifact_manifest"],
            "smoke": record["artifact_smoke"],
        },
        "dry_run": record["dry_run"],
    }


def _launch_full_training(
    *,
    level: RampLevelSpec,
    config_path: Path,
    run_dir: Path,
    device: str,
    reporter: TrainingReporter,
    resume_policy: str,
    resume_stage_checkpoints: Mapping[str, Path],
    gallery_output_root: Path | None,
    gallery_samples_per_group: int | None,
) -> dict[str, Any]:
    started = time.monotonic()
    settings = load_settings(config_path)
    result = run_full_training_pipeline(
        settings,
        config=FullTrainingRunConfig(
            run_dir=run_dir,
            device=torch.device(device),
            spatial_max_steps=level.spatial_steps,
            temporal_max_steps=level.temporal_steps,
            patch_limit=level.patch_limit,
            cache_max_frames=level.cache_frames,
            sequence_length=level.sequence_length,
            candidate_slots=level.candidate_slots,
            parameter_group_id="full-model",
            render_gallery=True,
            gallery_output_root=gallery_output_root,
            gallery_samples_per_group=(
                gallery_samples_per_group or level.gallery_samples_per_group
            ),
            reporter=reporter,
            resume_policy=resume_policy,
            resume_stage_checkpoints=resume_stage_checkpoints,
        ),
    )
    elapsed = time.monotonic() - started
    return {
        "status": "finished",
        "run_dir": str(run_dir),
        "elapsed_seconds": elapsed,
        "summary": result.as_summary(),
    }


def _write_final_readiness(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    target: RampTarget,
    levels: list[RampLevelSpec],
    auto_launch_full: bool,
    failure: str | None = None,
    status: str | None = None,
    updated_at_utc: str | None = None,
) -> Path:
    path = output_dir / "final_readiness.md"
    json_path = output_dir / "final_readiness.json"
    passed_levels = [
        level for level in manifest.get("levels", ()) if level.get("status") == "passed"
    ]
    allowed = failure is None and len(passed_levels) == len(levels)
    readiness_status = status or ("ready" if allowed else "blocked")
    readiness_updated_at = updated_at_utc or datetime.now(UTC).isoformat()
    command = _full_command_text(output_dir / "resolved_target_config.yaml", target)
    lines = [
        "# Training Ramp Final Readiness",
        "",
        f"status: {readiness_status}",
        f"updated_at_utc: {readiness_updated_at}",
        f"auto_launch_full: {auto_launch_full}",
        f"target: {target.__dict__}",
        "",
        "## Levels",
    ]
    for item in passed_levels:
        level = item["level"]
        lines.append(
            "- "
            f"{level['key']}: spatial={level['spatial_steps']} temporal={level['temporal_steps']} "
            f"frames={level['cache_frames']} peak_vram={item['peak_vram_gib']:.3f}GiB "
            f"steps/s={item['steps_per_second']:.3f} score={item['evaluation']['quality_score']}"
        )
    if failure:
        lines.extend(["", "## Blocking Failure", failure])
    lines.extend(
        [
            "",
            "## Slider Data Quality",
            json.dumps(
                manifest.get("preflight", {})
                .get("data", {})
                .get("slider_quality_issues", ()),
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "## Final Command",
            "```bash",
            command,
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(
        json_path,
        {
            "ready": allowed,
            "status": readiness_status,
            "updated_at_utc": readiness_updated_at,
            "failure": failure,
            "target": target.__dict__,
            "levels_passed": len(passed_levels),
            "levels_expected": len(levels),
            "full_command": command,
        },
    )
    (output_dir / "full_train_command.sh").write_text(command + "\n", encoding="utf-8")
    return path


def _run_job_dry_run(
    *,
    job_path: Path | None,
    config_path: Path,
    level_dir: Path,
    device: str,
) -> dict[str, Any]:
    if job_path is None:
        return {"returncode": 1, "error": "missing next job path"}
    log_path = level_dir / "logs" / "run_job_dry_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "traning.main",
            "run-job",
            "--job",
            str(job_path),
            "--config",
            str(config_path),
            "--device",
            device,
            "--dry-run",
        ],
        cwd=Path.cwd(),
        env={
            **os.environ,
            "PYTHONPATH": _pythonpath_with_src(),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {"returncode": completed.returncode, "log": str(log_path)}


def _pythonpath_with_src() -> str:
    entries = [str(Path.cwd() / "src"), str(Path.cwd())]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _write_level_config(
    base_config: Path,
    level_dir: Path,
    level: RampLevelSpec,
) -> Path:
    raw = _absolutize_config(_read_yaml(base_config), base_config.parent)
    raw["optimization"] = {
        **dict(raw.get("optimization") or {}),
        "enabled": True,
        "trial_store_path": str((level_dir / "metrics" / "trials.jsonl").resolve()),
        "trial_store_sqlite_path": str(
            (level_dir / "metrics" / "trials.sqlite").resolve()
        ),
    }
    raw["training_ramp_level"] = level.as_dict()
    path = level_dir / "resolved_level_config.yaml"
    _write_yaml(path, raw)
    return path


def _build_default_full_config(source: dict[str, Any]) -> dict[str, Any]:
    raw = json.loads(json.dumps(source, default=str))
    raw.setdefault("training_ramp", {})
    raw["training_ramp"]["target"] = RampTarget().__dict__
    raw.setdefault("candidate_cache", {})
    raw["candidate_cache"]["local_refiner_enabled"] = True
    raw["candidate_cache"]["ambiguity_review_enabled"] = True
    raw.setdefault("training", {})
    raw["training"].setdefault("spatial_consistency_loss_weights", {})
    raw["training"]["spatial_consistency_loss_weights"].update(
        {"embedding": 0.01, "ring_radius": 0.01, "slider_continuity": 0.01}
    )
    return raw


def _target_from_raw(raw: dict[str, Any]) -> RampTarget:
    target = (raw.get("training_ramp") or {}).get("target") or {}
    return RampTarget(
        spatial_steps=int(target.get("spatial_steps", RampTarget.spatial_steps)),
        temporal_steps=int(target.get("temporal_steps", RampTarget.temporal_steps)),
        patch_limit=int(target.get("patch_limit", RampTarget.patch_limit)),
        cache_frames=int(target.get("cache_frames", RampTarget.cache_frames)),
        sequence_length=int(target.get("sequence_length", RampTarget.sequence_length)),
        candidate_slots=int(target.get("candidate_slots", RampTarget.candidate_slots)),
        gallery_samples_per_group=int(
            target.get(
                "gallery_samples_per_group",
                RampTarget.gallery_samples_per_group,
            )
        ),
    )


def _clip_level(level: RampLevelSpec, target: RampTarget) -> RampLevelSpec:
    return RampLevelSpec(
        key=level.key,
        label=level.label,
        spatial_steps=min(level.spatial_steps, target.spatial_steps),
        temporal_steps=min(level.temporal_steps, target.temporal_steps),
        patch_limit=min(level.patch_limit, target.patch_limit),
        cache_frames=min(level.cache_frames, target.cache_frames),
        sequence_length=min(level.sequence_length, target.sequence_length),
        candidate_slots=min(level.candidate_slots, target.candidate_slots),
        gallery_samples_per_group=min(
            level.gallery_samples_per_group,
            target.gallery_samples_per_group,
        ),
    )


def _level_reaches_target(level: RampLevelSpec, target: RampTarget) -> bool:
    return (
        level.spatial_steps >= target.spatial_steps
        and level.temporal_steps >= target.temporal_steps
        and level.patch_limit >= target.patch_limit
        and level.cache_frames >= target.cache_frames
        and level.sequence_length >= target.sequence_length
        and level.candidate_slots >= target.candidate_slots
    )


def _init_layout(output_dir: Path) -> None:
    for child in ("logs", "checkpoints", "galleries", "metrics"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def _full_command_text(config_path: Path, target: RampTarget) -> str:
    return (
        "PYTHONPATH=src:. python -m traning.main run "
        f"--config {config_path} --device cuda "
        f"--spatial-max-steps {target.spatial_steps} "
        f"--temporal-max-steps {target.temporal_steps} "
        f"--patch-limit {target.patch_limit} "
        f"--cache-max-frames {target.cache_frames} "
        f"--sequence-length {target.sequence_length} "
        f"--candidate-slots {target.candidate_slots} "
        "--render-gallery "
        f"--gallery-samples-per-group {target.gallery_samples_per_group}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return raw


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_json_ready(value), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _absolutize_config(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    data = json.loads(json.dumps(raw, default=str))
    path_fields = (
        ("data_input", "dataset_root"),
        ("data_input", "split_manifest_path"),
        ("candidate_cache", "output_root"),
        ("visualization", "output_dir"),
        ("optimization", "trial_store_path"),
    )
    for section, key in path_fields:
        group = data.get(section)
        if not isinstance(group, dict) or group.get(key) in (None, ""):
            continue
        path = Path(str(group[key])).expanduser()
        group[key] = str(path if path.is_absolute() else (base_dir / path).resolve())
    return data


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


__all__ = [
    "DEFAULT_FULL_CONFIG",
    "DEFAULT_OUTPUT_ROOT",
    "RampGateError",
    "RampLevelSpec",
    "RampRunResult",
    "RampTarget",
    "build_ramp_levels",
    "ensure_full_target_config",
    "run_training_ramp",
]
