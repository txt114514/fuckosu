from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import hashlib
import json
import shutil
from typing import Any, Literal, Mapping

import torch

from start.flow import StartupFlowConfig, run_startup_flow
from start.samples import DEFAULT_MATCHED_MANIFEST
from traning.conf import DataSplit, load_settings
from traning.core.full_flow.result import FullFlowResult, FullFlowStageState, utc_now
from traning.core.full_flow.stages import (
    CRITICAL_STAGE_IDS,
    FULL_FLOW_STAGES,
    validate_stage_id,
)
from traning.core.model_export import ModelArtifactSpec, export_model_artifact
from traning.core.training_inheritance import (
    create_inheritance_package,
    load_inheritance_package,
)
from traning.core.training_ramp import run_training_ramp
from traning.state.versioning import collect_code_version, version_manifest
from visualization.lib import (
    PipelinePhase,
    PipelineStageState,
    TrainingEvent,
    TrainingReporter,
    TrainingStopState,
    collect_resource_state,
    create_dashboard_reporter,
)


FullFlowMode = Literal["execute", "plan", "dry-run", "status"]
FULL_FLOW_SCHEMA_VERSION = "full-flow-v1"
DEFAULT_FULL_FLOW_ROOT = Path("artifacts") / "training_runs"


@dataclass(frozen=True)
class FullFlowConfig:
    config_path: Path
    device: str = "auto"
    mode: FullFlowMode = "execute"
    output_root: Path = DEFAULT_FULL_FLOW_ROOT
    target_config_path: Path | None = None
    run_id: str | None = None
    auto_launch_full: bool = False
    force_level: bool = False
    max_levels: int | None = None
    run_full_checks: bool = True
    progress_ui: str = "auto"
    progress_language: str = "zh-CN"
    inherit_from: Path | str | None = None
    resume_policy: str = "none"
    resume_requested: bool = False
    before_config: Path | None = None
    split: DataSplit = "train"
    matched_manifest_path: Path = DEFAULT_MATCHED_MANIFEST
    skip_before_traning: bool = False
    before_match_probe: bool = True
    before_min_match_score: float = 0.1
    split_manifest: Path | None = None
    split_seed: int = 2026
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    allow_test_growth: bool = False
    test_level: str = "quick"
    gallery_output_root: Path | None = None
    gallery_samples_per_group: int | None = None
    from_stage: str | None = None
    until_stage: str | None = None
    force_stages: tuple[str, ...] = ()
    skip_stages: tuple[str, ...] = ()


@dataclass
class _FlowRuntime:
    config: FullFlowConfig
    run_id: str
    output_dir: Path
    started_at: str
    stages: dict[str, FullFlowStageState] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    resume_report: dict[str, Any] | None = None
    ramp_manifest_path: Path | None = None
    final_readiness_path: Path | None = None
    final_artifact_path: Path | None = None
    inheritance_path: Path | None = None
    reporter: TrainingReporter | None = None

    @property
    def state_path(self) -> Path:
        return self.output_dir / "full_flow_state.json"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "full_flow_manifest.json"

    @property
    def report_json_path(self) -> Path:
        return self.output_dir / "reports" / "full_flow_report.json"

    @property
    def report_markdown_path(self) -> Path:
        return self.output_dir / "reports" / "full_flow_report.md"


def run_full_flow(config: FullFlowConfig) -> FullFlowResult:
    _validate_config(config)
    run_id = config.run_id or _new_run_id()
    output_dir = config.output_root / run_id
    _init_layout(output_dir)
    runtime = _FlowRuntime(
        config=config,
        run_id=run_id,
        output_dir=output_dir,
        started_at=utc_now(),
        stages=_initial_stage_states(),
    )
    if config.mode == "status":
        return load_full_flow_status(config.output_root, run_id=config.run_id)
    _write_resolved_config(config.config_path, output_dir)
    runtime.manifest = _base_manifest(config, runtime)
    _persist(runtime, status="running")

    if config.mode == "plan":
        _mark_plan(runtime)
        _persist(runtime, status="planned")
        return _result(runtime, status="planned")

    with create_dashboard_reporter(
        run_id=run_id,
        output_dir=output_dir / "dashboard",
        progress_ui=config.progress_ui,
        progress_language=config.progress_language,
    ) as dashboard:
        reporter = dashboard.reporter
        runtime.reporter = reporter
        reporter.update_metrics(
            pipeline_phase=PipelinePhase.STARTUP.value,
            phase="完整流程启动检查",
            status="running",
            trial_status=None,
        )
        _report_resource_snapshot(reporter)
        _publish_initial_dashboard_stages(runtime)
        reporter.emit_event(
            TrainingEvent.create(
                event_type="full_flow",
                severity="info",
                message_key="dashboard_started",
                message_args={"run_id": run_id},
                raw_message=f"完整训练流程启动：{run_id}",
            )
        )
        try:
            _run_startup_section(runtime, reporter=reporter)
            inheritance = _run_resume_section(runtime, reporter=reporter)
            if config.mode == "dry-run":
                _mark_training_skipped_for_dry_run(runtime)
                _persist(runtime, status="dry-run-passed")
                return _result(runtime, status="dry-run-passed")
            _run_ramp_section(runtime, inheritance=inheritance, reporter=reporter)
            _run_finalize_section(runtime)
            _persist(runtime, status="passed")
            return _result(runtime, status="passed")
        except KeyboardInterrupt:
            _mark_interrupted(runtime, "用户中断完整流程")
            _persist(runtime, status="interrupted", stop_reason="用户中断完整流程")
            reporter.request_stop(
                TrainingStopState(
                    reason="USER_INTERRUPTED",
                    message="用户中断完整流程",
                    exit_code=2,
                )
            )
            raise
        except Exception as error:
            _mark_failed(runtime, error)
            _persist(runtime, status="failed", stop_reason=str(error))
            reporter.request_stop(
                TrainingStopState(
                    reason="FULL_FLOW_FAILED",
                    message=f"{type(error).__name__}: {error}",
                    exit_code=1,
                )
            )
            raise


def load_full_flow_status(
    output_root: Path = DEFAULT_FULL_FLOW_ROOT,
    *,
    run_id: str | None = None,
) -> FullFlowResult:
    pointer = output_root / "latest_full_flow.json"
    if run_id is None:
        if not pointer.exists():
            raise FileNotFoundError("full-flow status not found")
        raw_pointer = _read_json(pointer)
        state_path = Path(str(raw_pointer["state_path"]))
    else:
        state_path = output_root / run_id / "full_flow_state.json"
    raw = _read_json(state_path)
    stages = tuple(
        FullFlowStageState(
            stage_id=item["stage_id"],
            display_name=item["display_name"],
            status=item["status"],
            started_at=item.get("started_at"),
            ended_at=item.get("ended_at"),
            result=item.get("result") or {},
            warnings=tuple(item.get("warnings") or ()),
            error=item.get("error"),
            artifacts=tuple(item.get("artifacts") or ()),
            restored=bool(item.get("restored", False)),
        )
        for item in raw.get("stages", ())
    )
    return FullFlowResult(
        run_id=str(raw["run_id"]),
        mode=str(raw.get("mode", "status")),
        status=str(raw.get("status", "unknown")),
        output_dir=Path(str(raw["output_dir"])),
        manifest_path=Path(str(raw["manifest_path"])),
        state_path=state_path,
        report_json_path=Path(str(raw["report_json_path"])),
        report_markdown_path=Path(str(raw["report_markdown_path"])),
        stages=stages,
        started_at=str(raw.get("started_at") or ""),
        ended_at=raw.get("ended_at"),
        resume_report_path=(
            Path(str(raw["resume_report_path"]))
            if raw.get("resume_report_path")
            else None
        ),
        ramp_manifest_path=(
            Path(str(raw["ramp_manifest_path"]))
            if raw.get("ramp_manifest_path")
            else None
        ),
        final_readiness_path=(
            Path(str(raw["final_readiness_path"]))
            if raw.get("final_readiness_path")
            else None
        ),
        stop_reason=raw.get("stop_reason"),
    )


def _run_startup_section(_runtime: _FlowRuntime, *, reporter) -> None:
    config = _runtime.config
    selected_device = _select_device_name(config.device)
    _start_stage(_runtime, "ENVIRONMENT_PREFLIGHT", reporter)
    _start_stage(_runtime, "SOURCE_CHANGE_CHECK", reporter)
    _start_stage(_runtime, "BEFORE_TRAINING", reporter)
    _start_stage(_runtime, "DATASET_CONVERSION", reporter)
    _start_stage(_runtime, "SPLIT_VALIDATION", reporter)
    _start_stage(_runtime, "DATA_QUALITY_CHECK", reporter)
    startup = run_startup_flow(
        StartupFlowConfig(
            training_config=config.config_path,
            before_config=config.before_config,
            split=config.split,
            device=torch.device(selected_device),
            require_cuda=selected_device == "cuda",
            matched_manifest_path=config.matched_manifest_path,
            run_before_traning=not config.skip_before_traning,
            before_match_probe=config.before_match_probe,
            before_min_match_score=config.before_min_match_score,
            split_manifest_path=config.split_manifest,
            split_seed=config.split_seed,
            train_ratio=config.train_ratio,
            validation_ratio=config.validation_ratio,
            test_ratio=config.test_ratio,
            allow_test_growth=config.allow_test_growth,
            test_level=config.test_level,
            dry_run=config.mode == "dry-run",
            run_full_training=False,
            run_dir=_runtime.output_dir / "startup_full_training_unused",
            reporter=reporter,
        )
    )
    startup_path = _runtime.output_dir / "reports" / "startup_flow_report.json"
    _write_json(startup_path, startup.as_dict())
    _finish_stage(
        _runtime,
        "ENVIRONMENT_PREFLIGHT",
        "PASSED",
        result=startup.tests.as_dict(),
        artifacts=(str(startup_path),),
    )
    _finish_stage(
        _runtime,
        "SOURCE_CHANGE_CHECK",
        "PASSED",
        result=startup.before_startup.as_dict(),
        artifacts=(str(startup_path),),
    )
    _finish_stage(
        _runtime,
        "BEFORE_TRAINING",
        "PASSED" if startup.before_run.status == "passed" else "SKIPPED",
        result=startup.before_run.as_dict(),
        artifacts=(str(startup_path),),
    )
    _finish_stage(
        _runtime,
        "DATASET_CONVERSION",
        "PASSED" if startup.before_run.status == "passed" else "SKIPPED",
        result=startup.before_run.as_dict(),
        artifacts=(str(startup_path),),
    )
    _finish_stage(
        _runtime,
        "SPLIT_VALIDATION",
        "PASSED",
        result=startup.split_sync.as_dict(),
        artifacts=(str(startup.split_sync.manifest_path), str(startup_path)),
    )
    _finish_stage(
        _runtime,
        "DATA_QUALITY_CHECK",
        "PASSED",
        result=startup.training_startup.as_dict(),
        warnings=tuple(startup.training_startup.warnings),
        artifacts=(str(startup_path),),
    )
    _persist(_runtime, status="startup-passed")


def _run_resume_section(_runtime: _FlowRuntime, *, reporter):
    config = _runtime.config
    settings = load_settings(config.config_path)
    selected_inherit_from = (
        "latest"
        if config.resume_requested and config.inherit_from is None
        else config.inherit_from
    )
    selected_policy = (
        "auto"
        if config.resume_requested and config.resume_policy == "none"
        else config.resume_policy
    )
    _start_stage(_runtime, "RESUME_DISCOVERY", reporter)
    inheritance = load_inheritance_package(
        inherit_from=selected_inherit_from,
        current_settings=settings,
        policy=selected_policy,  # type: ignore[arg-type]
    )
    resume_report = inheritance.as_dict()
    resume_path = _runtime.output_dir / "resume_report.json"
    _write_json(resume_path, resume_report)
    _runtime.resume_report = resume_report
    _finish_stage(
        _runtime,
        "RESUME_DISCOVERY",
        "PASSED" if inheritance.compatible else "WARNING",
        result=resume_report,
        warnings=tuple(inheritance.downgrade_reasons),
        artifacts=(str(resume_path),),
    )
    _start_stage(_runtime, "RESUME_RESTORE", reporter)
    restored = tuple(inheritance.restored_fields)
    _finish_stage(
        _runtime,
        "RESUME_RESTORE",
        "RESTORED" if restored else "SKIPPED",
        result={
            "effective_policy": inheritance.policy,
            "stage_checkpoint_paths": inheritance.stage_checkpoint_paths,
            "restored_fields": restored,
            "missing_fields": inheritance.missing_fields,
        },
        warnings=tuple(inheritance.missing_fields),
        artifacts=tuple(str(path) for path in inheritance.stage_checkpoint_paths.values()),
        restored=bool(restored),
    )
    _persist(_runtime, status="resume-checked")
    return inheritance


def _run_ramp_section(_runtime: _FlowRuntime, *, inheritance, reporter) -> None:
    config = _runtime.config
    if not _stage_enabled(_runtime, "RAMP_TRAINING"):
        return
    _start_stage(_runtime, "RAMP_TRAINING", reporter)
    ramp = run_training_ramp(
        config_path=config.config_path,
        device=_select_device_name(config.device),
        output_root=_runtime.output_dir / "ramp",
        target_config_path=config.target_config_path,
        run_id="ramp",
        auto_launch_full=config.auto_launch_full,
        force_level=config.force_level or _stage_forced(config, "RAMP_TRAINING"),
        max_levels=config.max_levels,
        run_full_checks=config.run_full_checks,
        progress_ui="off",
        progress_language=config.progress_language,
        resume_policy=inheritance.policy,
        resume_stage_checkpoints=inheritance.stage_checkpoint_paths,
        full_gallery_output_root=config.gallery_output_root,
        full_gallery_samples_per_group=config.gallery_samples_per_group,
        reporter=reporter,
    )
    _runtime.ramp_manifest_path = ramp.manifest_path
    _runtime.final_readiness_path = ramp.final_readiness_path
    _finish_stage(
        _runtime,
        "RAMP_TRAINING",
        "PASSED",
        result=ramp.as_dict(),
        artifacts=(str(ramp.manifest_path),),
    )
    _start_stage(_runtime, "FINAL_READINESS", reporter)
    _finish_stage(
        _runtime,
        "FINAL_READINESS",
        "PASSED",
        result={"path": ramp.final_readiness_path},
        artifacts=(str(ramp.final_readiness_path),),
    )
    full_status = "PASSED" if ramp.full_training_started else "SKIPPED"
    _start_stage(_runtime, "FULL_TRAINING", reporter)
    _finish_stage(
        _runtime,
        "FULL_TRAINING",
        full_status,
        result={"run_dir": ramp.full_training_run_dir},
        artifacts=(
            (str(ramp.full_training_run_dir),)
            if ramp.full_training_run_dir is not None
            else ()
        ),
    )
    _start_stage(_runtime, "FINAL_EVALUATION", reporter)
    _finish_stage(
        _runtime,
        "FINAL_EVALUATION",
        "PASSED",
        result=_last_training_record(ramp.manifest_path) or {},
        artifacts=(str(ramp.manifest_path),),
    )
    _persist(_runtime, status="training-passed")


def _run_finalize_section(_runtime: _FlowRuntime) -> None:
    settings = load_settings(_runtime.config.config_path)
    record = _last_training_record(_runtime.ramp_manifest_path)
    _finish_export_stage(_runtime, settings=settings, record=record)
    _finish_inheritance_stage(_runtime, settings=settings, record=record)


def _finish_export_stage(
    _runtime: _FlowRuntime,
    *,
    settings,
    record: Mapping[str, Any] | None,
) -> None:
    stage = _runtime.stages["MODEL_EXPORT"]
    stage.mark_started()
    _report_full_flow_stage(_runtime, "MODEL_EXPORT")
    if not record:
        stage.mark_finished("SKIPPED", result={"reason": "no training record"})
        _report_full_flow_stage(_runtime, "MODEL_EXPORT")
        return
    spatial_path = _record_path(record, ("spatial", "checkpoint_path"))
    temporal_path = _record_path(record, ("temporal", "checkpoint_path"))
    if spatial_path is None or temporal_path is None:
        stage.mark_finished(
            "WARNING",
            result={"reason": "missing checkpoint path in training record"},
        )
        _report_full_flow_stage(_runtime, "MODEL_EXPORT")
        return
    artifact = export_model_artifact(
        ModelArtifactSpec(
            artifact_id=f"{_runtime.run_id}-final",
            output_dir=_runtime.output_dir / "artifacts" / "final_model",
            settings_path=_runtime.output_dir / "resolved_config.yaml",
            spatial_checkpoint_path=spatial_path,
            temporal_checkpoint_path=temporal_path,
            score_version="point-slider-v2+click-sequence-v1+aggregate-v1",
            candidate_cache_version="spatial-candidate-cache-v1",
            code_version=collect_code_version().commit,
            extra_files=_record_extra_files(record),
        )
    )
    _runtime.final_artifact_path = artifact.manifest_path
    stage.mark_finished(
        "PASSED",
        result={"artifact_manifest": artifact.manifest_path},
        artifacts=(str(artifact.manifest_path),),
    )
    _report_full_flow_stage(_runtime, "MODEL_EXPORT")


def _finish_inheritance_stage(
    _runtime: _FlowRuntime,
    *,
    settings,
    record: Mapping[str, Any] | None,
) -> None:
    stage = _runtime.stages["INHERITANCE_FINALIZATION"]
    stage.mark_started()
    _report_full_flow_stage(_runtime, "INHERITANCE_FINALIZATION")
    if not record:
        stage.mark_finished("SKIPPED", result={"reason": "no training record"})
        _report_full_flow_stage(_runtime, "INHERITANCE_FINALIZATION")
        return
    spatial_path = _record_path(record, ("spatial", "checkpoint_path"))
    temporal_path = _record_path(record, ("temporal", "checkpoint_path"))
    if temporal_path is None:
        stage.mark_finished("WARNING", result={"reason": "missing temporal checkpoint"})
        _report_full_flow_stage(_runtime, "INHERITANCE_FINALIZATION")
        return
    package = create_inheritance_package(
        output_dir=_runtime.output_dir,
        settings=settings,
        resolved_config_path=_runtime.output_dir / "resolved_config.yaml",
        latest_checkpoint_path=temporal_path,
        best_checkpoint_path=temporal_path,
        stage_checkpoints={"spatial": spatial_path, "temporal": temporal_path},
        training_state=dict(record),
        score_state=dict(record.get("evaluation") or {}),
        artifacts={
            "model_artifact": _runtime.final_artifact_path,
            **_record_extra_files(record),
        },
    )
    _runtime.inheritance_path = package.path
    stage.mark_finished(
        "PASSED",
        result=package.as_dict(),
        artifacts=(str(package.manifest_path),),
    )
    _report_full_flow_stage(_runtime, "INHERITANCE_FINALIZATION")


def _mark_plan(_runtime: _FlowRuntime) -> None:
    enabled = set(_selected_stage_ids(_runtime.config))
    for stage_id, state in _runtime.stages.items():
        if stage_id in enabled:
            state.mark_finished(
                "READY",
                result={
                    "mode": "plan",
                    "config": str(_runtime.config.config_path),
                    "output_dir": str(_runtime.output_dir),
                    "forced": _stage_forced(_runtime.config, stage_id),
                },
            )
        else:
            state.mark_finished("LOCKED", result={"reason": "outside stage range"})
        _report_full_flow_stage(_runtime, stage_id)


def _mark_training_skipped_for_dry_run(_runtime: _FlowRuntime) -> None:
    for stage_id in (
        "RAMP_TRAINING",
        "FINAL_READINESS",
        "FULL_TRAINING",
        "FINAL_EVALUATION",
        "MODEL_EXPORT",
        "INHERITANCE_FINALIZATION",
    ):
        _runtime.stages[stage_id].mark_finished(
            "SKIPPED",
            result={"reason": "dry-run does not execute training or checkpoint writes"},
        )
        _report_full_flow_stage(_runtime, stage_id)


def _mark_failed(_runtime: _FlowRuntime, error: Exception) -> None:
    for state in _runtime.stages.values():
        if state.status == "RUNNING":
            state.mark_finished(
                "FAILED",
                error=f"{type(error).__name__}: {error}",
            )
            _report_full_flow_stage(_runtime, state.stage_id)


def _mark_interrupted(_runtime: _FlowRuntime, reason: str) -> None:
    for state in _runtime.stages.values():
        if state.status == "RUNNING":
            state.mark_finished("INTERRUPTED", error=reason)
            _report_full_flow_stage(_runtime, state.stage_id)


def _persist(
    _runtime: _FlowRuntime,
    *,
    status: str,
    stop_reason: str | None = None,
) -> None:
    if status not in {
        "running",
        "startup-passed",
        "resume-checked",
        "training-passed",
    }:
        report_stage = _runtime.stages["REPORT_GENERATION"]
        if report_stage.status in {"PENDING", "RUNNING"}:
            if report_stage.status == "PENDING":
                report_stage.mark_started()
                _report_full_flow_stage(_runtime, "REPORT_GENERATION")
            report_stage.mark_finished(
                "PASSED",
                result={
                    "json": _runtime.report_json_path,
                    "markdown": _runtime.report_markdown_path,
                },
                artifacts=(
                    str(_runtime.report_json_path),
                    str(_runtime.report_markdown_path),
                ),
            )
            _report_full_flow_stage(_runtime, "REPORT_GENERATION")
    state = {
        "schema_version": FULL_FLOW_SCHEMA_VERSION,
        "run_id": _runtime.run_id,
        "mode": _runtime.config.mode,
        "status": status,
        "output_dir": _runtime.output_dir,
        "manifest_path": _runtime.manifest_path,
        "state_path": _runtime.state_path,
        "report_json_path": _runtime.report_json_path,
        "report_markdown_path": _runtime.report_markdown_path,
        "started_at": _runtime.started_at,
        "ended_at": utc_now() if status not in {"running", "startup-passed", "resume-checked", "training-passed"} else None,
        "stages": tuple(stage.as_dict() for stage in _runtime.stages.values()),
        "resume_report_path": (
            _runtime.output_dir / "resume_report.json"
            if _runtime.resume_report is not None
            else None
        ),
        "ramp_manifest_path": _runtime.ramp_manifest_path,
        "final_readiness_path": _runtime.final_readiness_path,
        "stop_reason": stop_reason,
    }
    manifest = dict(_runtime.manifest)
    manifest.update(
        {
            "status": status,
            "state_path": _runtime.state_path,
            "report_json_path": _runtime.report_json_path,
            "report_markdown_path": _runtime.report_markdown_path,
            "ramp_manifest_path": _runtime.ramp_manifest_path,
            "final_readiness_path": _runtime.final_readiness_path,
            "final_artifact_path": _runtime.final_artifact_path,
            "inheritance_path": _runtime.inheritance_path,
        }
    )
    _write_json(_runtime.state_path, state)
    _write_json(_runtime.manifest_path, manifest)
    _write_reports(_runtime, state)
    _write_json(
        _runtime.config.output_root / "latest_full_flow.json",
        {
            "run_id": _runtime.run_id,
            "state_path": _runtime.state_path,
            "manifest_path": _runtime.manifest_path,
            "status": status,
        },
    )


def _result(_runtime: _FlowRuntime, *, status: str) -> FullFlowResult:
    return FullFlowResult(
        run_id=_runtime.run_id,
        mode=_runtime.config.mode,
        status=status,
        output_dir=_runtime.output_dir,
        manifest_path=_runtime.manifest_path,
        state_path=_runtime.state_path,
        report_json_path=_runtime.report_json_path,
        report_markdown_path=_runtime.report_markdown_path,
        stages=tuple(_runtime.stages.values()),
        started_at=_runtime.started_at,
        ended_at=utc_now(),
        resume_report_path=(
            _runtime.output_dir / "resume_report.json"
            if _runtime.resume_report is not None
            else None
        ),
        ramp_manifest_path=_runtime.ramp_manifest_path,
        final_readiness_path=_runtime.final_readiness_path,
    )


def _base_manifest(config: FullFlowConfig, _runtime: _FlowRuntime) -> dict[str, Any]:
    settings = load_settings(config.config_path)
    return {
        "schema_version": FULL_FLOW_SCHEMA_VERSION,
        "run_id": _runtime.run_id,
        "created_at": _runtime.started_at,
        "config_path": config.config_path,
        "resolved_config_path": _runtime.output_dir / "resolved_config.yaml",
        "device": config.device,
        "mode": config.mode,
        "auto_launch_full": config.auto_launch_full,
        "gallery_output_root": config.gallery_output_root,
        "gallery_samples_per_group": config.gallery_samples_per_group,
        "stage_specs": tuple(stage.as_dict() for stage in FULL_FLOW_STAGES),
        "selected_stages": _selected_stage_ids(config),
        "force_stages": tuple(validate_stage_id(stage) for stage in config.force_stages),
        "skip_stages": tuple(validate_stage_id(stage) for stage in config.skip_stages),
        "config_fingerprint": _file_sha256(config.config_path),
        "dataset_fingerprint": _dataset_fingerprint(settings),
        "versions": version_manifest(settings),
        "code_version": collect_code_version().as_dict(),
    }


def _initial_stage_states() -> dict[str, FullFlowStageState]:
    return {
        stage.stage_id: FullFlowStageState(stage.stage_id, stage.display_name)
        for stage in FULL_FLOW_STAGES
    }


def _publish_initial_dashboard_stages(_runtime: _FlowRuntime) -> None:
    reporter = _runtime.reporter
    if reporter is None:
        return
    for stage_id in _selected_stage_ids(_runtime.config):
        stage = _runtime.stages[stage_id]
        reporter.update_pipeline_stage(
            PipelineStageState(
                stage_id=stage.stage_id.lower(),
                name=stage.display_name,
                status="pending",
            )
        )
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.STARTUP.value,
        phase="完整流程启动检查",
        status="running",
    )


def _report_resource_snapshot(reporter: TrainingReporter) -> None:
    with suppress(Exception):
        reporter.report_resource(collect_resource_state())


def _start_stage(_runtime: _FlowRuntime, stage_id: str, reporter) -> None:
    if not _stage_enabled(_runtime, stage_id):
        _runtime.stages[stage_id].mark_finished(
            "LOCKED",
            result={"reason": "outside requested stage range"},
        )
        _report_full_flow_stage(_runtime, stage_id)
        return
    stage = _runtime.stages[stage_id]
    stage.mark_started()
    reporter.update_metrics(
        pipeline_phase=_phase_for_full_flow_stage(stage_id).value,
        phase=stage.display_name,
        status="running",
    )
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id=stage_id.lower(),
            name=stage.display_name,
            status="running",
        )
    )


def _finish_stage(
    _runtime: _FlowRuntime,
    stage_id: str,
    status,
    *,
    result: Mapping[str, Any] | None = None,
    warnings: tuple[str, ...] = (),
    artifacts: tuple[str, ...] = (),
    restored: bool = False,
) -> None:
    result = dict(result or {})
    if _stage_forced(_runtime.config, stage_id):
        result["forced"] = True
    _runtime.stages[stage_id].mark_finished(
        status,
        result=result,
        warnings=warnings,
        artifacts=artifacts,
        restored=restored,
    )
    _report_full_flow_stage(_runtime, stage_id)


def _report_full_flow_stage(_runtime: _FlowRuntime, stage_id: str) -> None:
    reporter = _runtime.reporter
    if reporter is None:
        return
    stage = _runtime.stages[stage_id]
    pipeline_phase = _phase_for_full_flow_stage(stage_id)
    if stage.status in {"FAILED", "INTERRUPTED"}:
        pipeline_phase = PipelinePhase.FAILED
    elif stage.stage_id == "REPORT_GENERATION" and stage.status in {"PASSED", "COMPLETED"}:
        pipeline_phase = PipelinePhase.COMPLETED
    reporter.update_metrics(
        pipeline_phase=pipeline_phase.value,
        phase=stage.display_name,
        status=_dashboard_status(stage.status),
    )
    result = dict(stage.result or {})
    processed = _optional_int(result.get("processed"), default=0)
    total = _optional_int(result.get("total"))
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id=stage.stage_id.lower(),
            name=stage.display_name,
            status=_dashboard_status(stage.status),
            processed=processed or 0,
            total=total,
            output_path=stage.artifacts[0] if stage.artifacts else None,
            warning_count=len(stage.warnings),
            error_reason=stage.error,
            blocks_training=stage.status in {"FAILED", "INTERRUPTED"},
            message=str(result.get("message")) if result.get("message") else None,
        )
    )


def _dashboard_status(status: str) -> str:
    return {
        "PENDING": "pending",
        "READY": "pending",
        "RUNNING": "running",
        "PASSED": "passed",
        "WARNING": "warning",
        "FAILED": "failed",
        "SKIPPED": "skipped",
        "INTERRUPTED": "interrupted",
        "RESTORED": "passed",
        "COMPLETED": "completed",
        "LOCKED": "skipped",
    }.get(status, status.lower())


def _phase_for_full_flow_stage(stage_id: str) -> PipelinePhase:
    if stage_id in {
        "SOURCE_CHANGE_CHECK",
        "BEFORE_TRAINING",
        "DATASET_CONVERSION",
        "SPLIT_VALIDATION",
    }:
        return PipelinePhase.DATA_PREPARATION
    if stage_id in {
        "DATA_QUALITY_CHECK",
        "ENVIRONMENT_PREFLIGHT",
        "RESUME_DISCOVERY",
        "RESUME_RESTORE",
    }:
        return PipelinePhase.PRETRAIN_CHECK
    if stage_id == "FINAL_READINESS":
        return PipelinePhase.PROGRESSIVE_PREPARATION
    if stage_id in {
        "RAMP_TRAINING",
        "FULL_TRAINING",
        "FINAL_EVALUATION",
        "MODEL_EXPORT",
        "INHERITANCE_FINALIZATION",
        "REPORT_GENERATION",
    }:
        return PipelinePhase.TRAINING
    return PipelinePhase.STARTUP


def _optional_int(value: object, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _stage_enabled(_runtime: _FlowRuntime, stage_id: str) -> bool:
    return stage_id in _selected_stage_ids(_runtime.config)


def _stage_forced(config: FullFlowConfig, stage_id: str) -> bool:
    return stage_id in {validate_stage_id(stage) for stage in config.force_stages}


def _selected_stage_ids(config: FullFlowConfig) -> tuple[str, ...]:
    ids = tuple(stage.stage_id for stage in FULL_FLOW_STAGES)
    start = ids.index(validate_stage_id(config.from_stage)) if config.from_stage else 0
    end = ids.index(validate_stage_id(config.until_stage)) + 1 if config.until_stage else len(ids)
    selected = list(ids[start:end])
    skipped = {validate_stage_id(stage) for stage in config.skip_stages}
    blocked = skipped & CRITICAL_STAGE_IDS
    if blocked:
        raise ValueError("cannot skip critical full-flow stages: " + ", ".join(sorted(blocked)))
    return tuple(stage for stage in selected if stage not in skipped)


def _validate_config(config: FullFlowConfig) -> None:
    if config.mode not in {"execute", "plan", "dry-run", "status"}:
        raise ValueError("full-flow mode must be execute, plan, dry-run, or status")
    if config.resume_policy not in {"strict", "auto", "weights-only", "none"}:
        raise ValueError("resume-policy must be strict, auto, weights-only, or none")
    if config.progress_ui not in {"auto", "rich", "plain", "off"}:
        raise ValueError("progress-ui must be auto, rich, plain, or off")
    selected = set(_selected_stage_ids(config))
    forced = {validate_stage_id(stage) for stage in config.force_stages}
    skipped = {validate_stage_id(stage) for stage in config.skip_stages}
    conflicting = forced & skipped
    if conflicting:
        raise ValueError("cannot force and skip the same full-flow stages: " + ", ".join(sorted(conflicting)))
    outside = forced - selected
    if outside:
        raise ValueError("forced full-flow stages must be inside the selected stage range: " + ", ".join(sorted(outside)))


def _init_layout(output_dir: Path) -> None:
    for child in (
        "logs",
        "checkpoints",
        "galleries",
        "artifacts",
        "exports",
        "inheritance",
        "reports",
        "dashboard",
        "ramp",
    ):
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def _write_resolved_config(config_path: Path, output_dir: Path) -> None:
    if config_path.exists():
        shutil.copy2(config_path, output_dir / "resolved_config.yaml")


def _write_reports(_runtime: _FlowRuntime, state: Mapping[str, Any]) -> None:
    _write_json(_runtime.report_json_path, dict(state))
    stage_lines = [
        f"- {stage.display_name} (`{stage.stage_id}`): {stage.status}"
        + (f" - {stage.error}" if stage.error else "")
        for stage in _runtime.stages.values()
    ]
    lines = [
        "# Full Flow Report",
        "",
        f"- run_id: `{_runtime.run_id}`",
        f"- mode: `{_runtime.config.mode}`",
        f"- status: `{state.get('status')}`",
        f"- started_at: `{_runtime.started_at}`",
        f"- output_dir: `{_runtime.output_dir}`",
        "",
        "## Stages",
        *stage_lines,
    ]
    if _runtime.ramp_manifest_path:
        lines.extend(["", f"- ramp_manifest: `{_runtime.ramp_manifest_path}`"])
    if _runtime.final_artifact_path:
        lines.append(f"- final_artifact: `{_runtime.final_artifact_path}`")
    if _runtime.inheritance_path:
        lines.append(f"- inheritance: `{_runtime.inheritance_path}`")
    _runtime.report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _runtime.report_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _last_training_record(ramp_manifest_path: Path | None) -> Mapping[str, Any] | None:
    if ramp_manifest_path is None or not ramp_manifest_path.exists():
        return None
    manifest = _read_json(ramp_manifest_path)
    if manifest.get("full_training"):
        summary = (manifest["full_training"] or {}).get("summary") or {}
        run_dir = Path(str((manifest["full_training"] or {}).get("run_dir")))
        return {
            "run_dir": str(run_dir),
            "spatial": {"checkpoint_path": str(run_dir / "spatial" / "spatial_model.pt")},
            "temporal": {"checkpoint_path": summary.get("temporal_checkpoint")},
            "evaluation": {
                "report_path": str(run_dir / "evaluation" / "trial_score_report.json"),
                "gallery_request_path": str(run_dir / "evaluation" / "gallery_request.json"),
                "next_job_path": str(run_dir / "evaluation" / "next_training_job.json"),
            },
            "summary": summary,
        }
    levels = [level for level in manifest.get("levels", ()) if level.get("status") == "passed"]
    return levels[-1] if levels else None


def _record_path(record: Mapping[str, Any], keys: tuple[str, ...]) -> Path | None:
    current: Any = record
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    if current in (None, ""):
        return None
    path = Path(str(current))
    return path if path.exists() else None


def _record_extra_files(record: Mapping[str, Any]) -> dict[str, Path]:
    candidates = {
        "score_report": _record_path(record, ("evaluation", "report_path")),
        "gallery_request": _record_path(record, ("evaluation", "gallery_request_path")),
        "summary": _record_path(record, ("summary_path",)),
        "candidate_cache_manifest": _record_path(record, ("candidate_cache", "manifest_path")),
    }
    return {key: path for key, path in candidates.items() if path is not None}


def _select_device_name(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available inside the container")
    return device


def _dataset_fingerprint(settings) -> dict[str, Any]:
    data = settings.data_input
    return {
        "dataset_root": str(data.dataset_root),
        "split_manifest_path": str(data.split_manifest_path),
        "train_items": tuple(data.train_items),
        "validation_items": tuple(data.validation_items),
        "test_items": tuple(data.test_items),
        "sample_fps": data.sample_fps,
        "frame_step": data.frame_step,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "DEFAULT_FULL_FLOW_ROOT",
    "FULL_FLOW_SCHEMA_VERSION",
    "FullFlowConfig",
    "FullFlowMode",
    "load_full_flow_status",
    "run_full_flow",
]
