from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

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
from traning.core.optimization import (
    DecisionOutputScoreResult,
    OptimizationExecutorConfig,
    ParameterSearchConfig,
    TrialScoreSpec,
    analyze_trial_attribution,
    build_batch_gallery_request,
    create_trial_store,
    execute_optimization_plan,
    plan_next_trial,
    score_decision_outputs,
)
from traning.lib.metrics import SequenceScoreSpec
from traning.state.versioning import version_manifest
from traning.core.result_export import save_annotation_gallery
from traning.core.spatial import SpatialTrainingResult, run_spatial_training
from traning.core.temporal import TemporalTrainingResult, run_temporal_training
from visualization.lib import (
    DatasetUsageState,
    NullReporter,
    PipelinePhase,
    PipelineStageState,
    ResourceState,
    TrainingEvent,
    TrainingReporter,
    collect_resource_state,
)


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
    parameter_group_id: str = "pg-0001"
    curriculum_level: str | None = None
    render_gallery: bool = True
    gallery_output_root: Path | None = None
    gallery_samples_per_group: int | None = None
    reporter: TrainingReporter = field(default_factory=NullReporter)
    resume_policy: str = "none"
    resume_stage_checkpoints: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parameter_group_id:
            raise ValueError("parameter_group_id must not be empty")
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
            "gallery_samples_per_group",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")


@dataclass(frozen=True)
class FullTrainingEvaluationResult:
    parameter_group_id: str
    quality_score: float
    pass_threshold: float
    passed: bool
    target_count: int
    hit_count: int
    miss_count: int
    unresolved_count: int
    frequency_limited_count: int
    candidate_frame_count: int
    decision_frame_count: int
    no_op_frame_count: int
    action_frame_count: int
    report_path: Path
    gallery_request_path: Path
    gallery_status: str
    gallery_output_dir: Path | None
    gallery_saved_frame_count: int
    attribution_path: Path | None = None
    optimization_plan_path: Path | None = None
    next_job_path: Path | None = None
    gallery_warning: str | None = None
    asha_action: str | None = None
    asha_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter_group_id": self.parameter_group_id,
            "quality_score": self.quality_score,
            "pass_threshold": self.pass_threshold,
            "passed": self.passed,
            "target_count": self.target_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "unresolved_count": self.unresolved_count,
            "frequency_limited_count": self.frequency_limited_count,
            "candidate_frame_count": self.candidate_frame_count,
            "decision_frame_count": self.decision_frame_count,
            "no_op_frame_count": self.no_op_frame_count,
            "action_frame_count": self.action_frame_count,
            "report_path": self.report_path,
            "gallery_request_path": self.gallery_request_path,
            "gallery_status": self.gallery_status,
            "gallery_output_dir": self.gallery_output_dir,
            "gallery_saved_frame_count": self.gallery_saved_frame_count,
            "attribution_path": self.attribution_path,
            "optimization_plan_path": self.optimization_plan_path,
            "next_job_path": self.next_job_path,
            "gallery_warning": self.gallery_warning,
            "asha_action": self.asha_action,
            "asha_reasons": list(self.asha_reasons),
        }


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
    evaluation: FullTrainingEvaluationResult

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
            "evaluation": self.evaluation.as_dict(),
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
            "parameter_group_id": self.evaluation.parameter_group_id,
            "quality_score": self.evaluation.quality_score,
            "quality_pass_threshold": self.evaluation.pass_threshold,
            "quality_passed": self.evaluation.passed,
            "asha_action": self.evaluation.asha_action,
            "asha_reasons": self.evaluation.asha_reasons,
            "score_targets": self.evaluation.target_count,
            "score_hits": self.evaluation.hit_count,
            "score_misses": self.evaluation.miss_count,
            "score_unresolved": self.evaluation.unresolved_count,
            "score_frequency_limited": self.evaluation.frequency_limited_count,
            "score_no_op_frames": self.evaluation.no_op_frame_count,
            "score_action_frames": self.evaluation.action_frame_count,
            "gallery_status": self.evaluation.gallery_status,
            "gallery_output": self.evaluation.gallery_output_dir,
            "gallery_saved_frames": self.evaluation.gallery_saved_frame_count,
        }


TRAINING_STAGES = (
    TrainingStage("data_input", "inspect configured training split"),
    TrainingStage("spatial", "train first-version spatial model"),
    TrainingStage("candidate_cache", "build spatial candidate cache"),
    TrainingStage("temporal", "train causal temporal decision model"),
    TrainingStage("decision", "export frame-level temporal decisions"),
    TrainingStage("evaluation", "score parameter group and export best gallery"),
)


def run_full_training_pipeline(
    settings: Settings,
    *,
    config: FullTrainingRunConfig,
) -> FullTrainingRunResult:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    reporter = config.reporter
    reporter.update_metrics(
        pipeline_phase=PipelinePhase.TRAINING.value,
        phase="完整训练流水线",
        status="running",
        current_trial_id=config.parameter_group_id,
        trial_status="training",
        current_level=config.curriculum_level or "full_training",
        current_grade="observing",
        global_step=0,
        target_global_steps=config.spatial_max_steps + config.temporal_max_steps,
        promotion_status="完整训练流水线启动",
        current_parameters=_training_parameter_config_snapshot(settings, config=config),
    )
    reporter.emit_event(
        TrainingEvent.create(
            event_type="training",
            severity="info",
            message_key="stage_started",
            message_args={"stage": "完整训练流水线"},
        )
    )
    _report_stage(reporter, "startup", "训练前检测", "checking")
    startup_checks = run_training_startup_checks(
        settings,
        split=config.split,
        device=config.device,
    )
    startup_checks.raise_for_errors()
    data_report = startup_checks.data_input
    _report_stage(
        reporter,
        "startup",
        "训练前检测",
        "passed",
        processed=len(startup_checks.report.results),
        total=len(startup_checks.report.results),
        warnings=len(startup_checks.report.warnings),
    )
    _report_stage(
        reporter,
        "data_quality",
        "数据质量检查",
        "warning" if data_report.issues else "passed",
        processed=data_report.segment_count,
        total=data_report.segment_count,
        warnings=data_report.issue_count,
        blocks_training=not data_report.ok,
    )
    reporter.report_dataset_usage(
        DatasetUsageState(
            total_segments=data_report.segment_count,
            total_frames=data_report.frame_count_estimate,
            unique_segments=0,
            unique_frames=0,
        )
    )

    _report_stage(
        reporter,
        "spatial",
        "空间训练",
        "running",
        total=config.spatial_max_steps,
    )
    spatial = run_spatial_training(
        settings,
        device=config.device,
        run_dir=config.run_dir / "spatial",
        split=config.split,
        max_steps=config.spatial_max_steps,
        learning_rate=config.spatial_learning_rate,
        patch_limit=config.patch_limit,
        reporter=reporter,
        resume_checkpoint_path=config.resume_stage_checkpoints.get("spatial"),
        resume_policy=config.resume_policy,
    )
    reporter.update_metrics(
        loss=float(spatial.last_loss),
        spatial_step=spatial.steps,
        spatial_target=config.spatial_max_steps,
    )
    reporter.register_checkpoint(spatial.checkpoint_path)
    _report_resource(reporter)
    _report_stage(
        reporter,
        "spatial",
        "空间训练",
        "passed",
        processed=spatial.steps,
        total=config.spatial_max_steps,
        output_path=spatial.checkpoint_path,
    )
    _report_stage(
        reporter,
        "candidate_cache",
        "候选缓存",
        "running",
        total=config.cache_max_frames,
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
    reporter.report_dataset_usage(
        DatasetUsageState(
            total_segments=data_report.segment_count,
            total_frames=data_report.frame_count_estimate,
            sampled_frames=candidate_cache.frames,
            unique_frames=candidate_cache.frames,
            cached_frames=candidate_cache.frames,
            generated_candidates=candidate_cache.candidates,
        )
    )
    _report_stage(
        reporter,
        "candidate_cache",
        "候选缓存",
        "passed",
        processed=candidate_cache.frames,
        total=config.cache_max_frames,
        output_path=candidate_cache.output_dir,
    )
    _report_stage(
        reporter,
        "temporal",
        "时序训练",
        "running",
        total=config.temporal_max_steps,
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
        reporter=reporter,
        resume_checkpoint_path=config.resume_stage_checkpoints.get("temporal"),
        resume_policy=config.resume_policy,
    )
    reporter.update_metrics(
        loss=float(temporal.final_loss),
        temporal_step=temporal.steps,
        temporal_target=config.temporal_max_steps,
    )
    reporter.register_checkpoint(temporal.checkpoint_path, is_best=True)
    _report_resource(reporter)
    _report_stage(
        reporter,
        "temporal",
        "时序训练",
        "passed",
        processed=temporal.steps,
        total=config.temporal_max_steps,
        output_path=temporal.checkpoint_path,
    )
    _report_stage(reporter, "decision", "决策导出", "running")
    decision = run_temporal_decision(
        settings,
        cache_dir=candidate_cache.output_dir,
        checkpoint_path=temporal.checkpoint_path,
        output_dir=config.run_dir / "decision",
        device=config.device,
    )
    _report_stage(
        reporter,
        "decision",
        "决策导出",
        "passed",
        processed=decision.frames,
        total=candidate_cache.frames,
        output_path=decision.output_dir,
    )
    _report_stage(reporter, "evaluation", "固定评估评分", "running")
    reporter.update_metrics(
        phase="固定评估评分",
        status="evaluating",
        trial_status="evaluating",
        promotion_status="正在评估当前参数",
    )
    evaluation = _evaluate_training_outputs(
        settings,
        config=config,
        candidate_cache=candidate_cache,
        spatial=spatial,
        temporal=temporal,
        decision=decision,
    )
    trial_status, current_grade, promotion_status, prune_reason = _trial_outcome(
        evaluation
    )
    reporter.update_metrics(
        phase="固定评估评分",
        status="passed" if evaluation.passed else "warning",
        score=float(evaluation.quality_score),
        current_grade=current_grade,
        best_grade=current_grade if evaluation.passed else None,
        trial_status=trial_status,
        prune_reason=prune_reason,
        promotion_status=promotion_status,
        consecutive_passes=1 if evaluation.passed else 0,
        required_passes=1,
        current_parameters=_full_training_parameter_snapshot(
            settings,
            config=config,
            spatial=spatial,
            candidate_cache=candidate_cache,
            temporal=temporal,
            decision=decision,
            evaluation=evaluation,
        )
    )
    reporter.report_score(
        score=float(evaluation.quality_score),
        trial_id=config.parameter_group_id,
        category_scores=_category_scores_from_report(evaluation.report_path),
    )
    if evaluation.gallery_output_dir is not None:
        reporter.emit_event(
            TrainingEvent.create(
                event_type="gallery",
                severity="success",
                message_key="gallery_saved",
                message_args={"path": str(evaluation.gallery_output_dir)},
            )
        )
    _report_stage(
        reporter,
        "evaluation",
        "固定评估评分",
        "passed" if evaluation.passed else "warning",
        processed=evaluation.decision_frame_count,
        total=evaluation.candidate_frame_count,
        output_path=evaluation.report_path,
        warnings=evaluation.unresolved_count,
        score=evaluation.quality_score,
        threshold=evaluation.pass_threshold,
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
        evaluation=evaluation,
    )
    summary_path.write_text(
        json.dumps(_json_ready(result.as_dict()), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    reporter.emit_event(
        TrainingEvent.create(
            event_type="training",
            severity="success",
            message_key="stage_finished",
            message_args={"stage": "完整训练流水线", "status": "已完成"},
        )
    )
    return result


def _evaluate_training_outputs(
    settings: Settings,
    *,
    config: FullTrainingRunConfig,
    candidate_cache: CandidateCacheBuildResult,
    spatial: SpatialTrainingResult,
    temporal: TemporalTrainingResult,
    decision: TemporalDecisionRunResult,
) -> FullTrainingEvaluationResult:
    output_dir = config.run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_result = score_decision_outputs(
        parameter_group_id=config.parameter_group_id,
        candidate_cache_path=candidate_cache.records_path,
        decisions_path=decision.decisions_path,
        settings=settings,
        spec=TrialScoreSpec(
            sequence_spec=SequenceScoreSpec(
                min_click_interval_ms=settings.evaluation.min_click_interval_ms
            )
        ),
        metrics={
            "spatial_last_loss": float(spatial.last_loss),
            "temporal_final_loss": float(temporal.final_loss),
            "temporal_action_loss": float(temporal.action_loss),
            "temporal_candidate_loss": float(temporal.candidate_loss),
            "temporal_xy_loss": float(temporal.xy_loss),
            "temporal_time_loss": float(temporal.time_loss),
            "min_click_interval_ms": float(
                settings.evaluation.min_click_interval_ms
            ),
        },
    )
    report_path = output_dir / "trial_score_report.json"
    versions = version_manifest(settings) | {
        "score_version": score_result.report.score_version,
        "candidate_cache_version": "spatial-candidate-cache-v1",
        "trial_id": config.parameter_group_id,
    }
    report_path.write_text(
        json.dumps(
            _json_ready(score_result.report.as_dict() | {"versions": versions}),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    gallery_request = build_batch_gallery_request(
        score_result.report,
        batch_id=f"{config.run_dir.name}__{config.parameter_group_id}",
        metadata=versions
        | {
            "min_click_interval_ms": settings.evaluation.min_click_interval_ms,
            "candidate_cache_path": str(candidate_cache.records_path),
            "decision_path": str(decision.decisions_path),
        },
    )
    gallery_request_path = output_dir / "gallery_request.json"
    gallery_request_path.write_text(
        json.dumps(
            gallery_request.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    gallery_result = None
    if config.render_gallery:
        gallery_result = save_annotation_gallery(
            settings,
            gallery_request,
            output_root=config.gallery_output_root,
            samples_per_group=config.gallery_samples_per_group,
        )
    attribution_path = None
    plan_path = None
    next_job_path = None
    asha_action = None
    asha_reasons: tuple[str, ...] = ()
    if settings.optimization.enabled:
        attribution = analyze_trial_attribution(score_result.report)
        plan = plan_next_trial(
            score_result.report,
            attribution,
            config=ParameterSearchConfig(
                objective_weights=settings.optimization.objective_weights
            ),
        )
        asha_action = plan.asha_action
        asha_reasons = plan.asha_reasons
        execution = execute_optimization_plan(
            score_result.report,
            attribution,
            plan,
            parent_checkpoint_path=temporal.checkpoint_path,
            config=OptimizationExecutorConfig(
                output_dir=settings.optimization.trial_store_path.parent,
                code_version=json.dumps(versions["code_version"], sort_keys=True),
                data_version=str(versions["dataset_version"]),
            ),
            store=create_trial_store(
                backend=settings.optimization.trial_store_backend,
                jsonl_path=settings.optimization.trial_store_path,
                sqlite_path=settings.optimization.trial_store_sqlite_path,
            ),
        )
        attribution_path = output_dir / "attribution.json"
        plan_path = output_dir / "optimization_plan.json"
        next_job_path = output_dir / "next_training_job.json"
        attribution_path.write_text(
            json.dumps(_json_ready(attribution.as_dict()), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        plan_path.write_text(
            json.dumps(_json_ready(plan.as_dict()), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        next_job_path.write_text(
            json.dumps(
                _json_ready(execution.job.as_dict()),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return _evaluation_result_from_score(
        score_result,
        report_path=report_path,
        gallery_request_path=gallery_request_path,
        gallery_status=(
            gallery_result.status if gallery_result is not None else "skipped"
        ),
        gallery_output_dir=(
            gallery_result.output_dir if gallery_result is not None else None
        ),
        gallery_saved_frame_count=(
            gallery_result.saved_frame_count if gallery_result is not None else 0
        ),
        attribution_path=attribution_path,
        optimization_plan_path=plan_path,
        next_job_path=next_job_path,
        gallery_warning=(
            gallery_result.warning if gallery_result is not None else None
        ),
        asha_action=asha_action,
        asha_reasons=asha_reasons,
    )


def _evaluation_result_from_score(
    score_result: DecisionOutputScoreResult,
    *,
    report_path: Path,
    gallery_request_path: Path,
    gallery_status: str,
    gallery_output_dir: Path | None,
    gallery_saved_frame_count: int,
    attribution_path: Path | None,
    optimization_plan_path: Path | None,
    next_job_path: Path | None,
    gallery_warning: str | None,
    asha_action: str | None,
    asha_reasons: tuple[str, ...],
) -> FullTrainingEvaluationResult:
    report = score_result.report
    return FullTrainingEvaluationResult(
        parameter_group_id=score_result.parameter_group_id,
        quality_score=report.quality_score,
        pass_threshold=report.pass_threshold,
        passed=report.passed,
        target_count=report.target_count,
        hit_count=report.hit_count,
        miss_count=report.miss_count,
        unresolved_count=report.unresolved_count,
        frequency_limited_count=report.frequency_limited_count,
        candidate_frame_count=score_result.candidate_frame_count,
        decision_frame_count=score_result.decision_frame_count,
        no_op_frame_count=score_result.no_op_frame_count,
        action_frame_count=score_result.action_frame_count,
        report_path=report_path,
        gallery_request_path=gallery_request_path,
        gallery_status=gallery_status,
        gallery_output_dir=gallery_output_dir,
        gallery_saved_frame_count=gallery_saved_frame_count,
        attribution_path=attribution_path,
        optimization_plan_path=optimization_plan_path,
        next_job_path=next_job_path,
        gallery_warning=gallery_warning,
        asha_action=asha_action,
        asha_reasons=asha_reasons,
    )


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
        "distribution": report.distribution,
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


def _full_training_parameter_snapshot(
    settings: Settings,
    *,
    config: FullTrainingRunConfig,
    spatial: SpatialTrainingResult,
    candidate_cache: CandidateCacheBuildResult,
    temporal: TemporalTrainingResult,
    decision: TemporalDecisionRunResult,
    evaluation: FullTrainingEvaluationResult,
) -> dict[str, Any]:
    snapshot = _training_parameter_config_snapshot(settings, config=config)
    snapshot.update(
        {
            "evaluation": {
                "min_click_interval_ms": settings.evaluation.min_click_interval_ms,
                "quality_score": evaluation.quality_score,
                "pass_threshold": evaluation.pass_threshold,
                "passed": evaluation.passed,
                "report_path": evaluation.report_path,
                "gallery_status": evaluation.gallery_status,
                "gallery_samples_per_group": config.gallery_samples_per_group,
                "asha_action": evaluation.asha_action,
                "asha_reasons": evaluation.asha_reasons,
            },
            "outputs": {
                "spatial_checkpoint": spatial.checkpoint_path,
                "temporal_checkpoint": temporal.checkpoint_path,
                "candidate_cache_records": candidate_cache.records_path,
                "decision_path": decision.decisions_path,
                "candidate_frames": candidate_cache.frames,
                "candidate_count": candidate_cache.candidates,
                "decision_frames": decision.frames,
            },
            "losses": {
                "spatial_last_loss": spatial.last_loss,
                "temporal_final_loss": temporal.final_loss,
                "temporal_action_loss": temporal.action_loss,
                "temporal_candidate_loss": temporal.candidate_loss,
                "temporal_xy_loss": temporal.xy_loss,
                "temporal_time_loss": temporal.time_loss,
            },
        }
    )
    return _json_ready(snapshot)


def _training_parameter_config_snapshot(
    settings: Settings,
    *,
    config: FullTrainingRunConfig,
) -> dict[str, Any]:
    return _json_ready(
        {
            "parameter_group_id": config.parameter_group_id,
            "device": str(config.device),
            "split": config.split,
            "curriculum_level": config.curriculum_level,
            "training": {
                "spatial_max_steps": config.spatial_max_steps,
                "temporal_max_steps": config.temporal_max_steps,
                "spatial_learning_rate": config.spatial_learning_rate,
                "temporal_learning_rate": config.temporal_learning_rate,
                "patch_limit": config.patch_limit,
                "resume_policy": config.resume_policy,
                "resume_stage_checkpoints": dict(config.resume_stage_checkpoints),
            },
            "candidate_cache": {
                "cache_max_frames": config.cache_max_frames,
                "max_candidates": config.max_candidates,
                "score_threshold": config.score_threshold,
                "nms_radius_px": config.nms_radius_px,
                "slider_threshold": config.slider_threshold,
                "max_slider_paths": config.max_slider_paths,
                "local_refiner_enabled": settings.candidate_cache.local_refiner_enabled,
                "ambiguity_review_enabled": (
                    settings.candidate_cache.ambiguity_review_enabled
                ),
            },
            "temporal": {
                "sequence_length": config.sequence_length,
                "candidate_slots": config.candidate_slots,
                "smet_enabled": settings.smet.enabled,
                "smet_sparsity": settings.smet.sparsity,
                "smet_update_interval": settings.smet.update_interval,
                "smet_min_density": settings.smet.min_density,
            },
            "evaluation": {
                "min_click_interval_ms": settings.evaluation.min_click_interval_ms,
                "quality_score": None,
                "pass_threshold": None,
                "passed": None,
                "report_path": None,
                "gallery_status": "pending" if config.render_gallery else "skipped",
                "gallery_samples_per_group": config.gallery_samples_per_group,
                "asha_action": None,
                "asha_reasons": (),
            },
            "optimization": {
                "enabled": settings.optimization.enabled,
                "trial_store_backend": settings.optimization.trial_store_backend,
                "objective_weights": settings.optimization.objective_weights,
            },
        }
    )


def _trial_outcome(
    evaluation: FullTrainingEvaluationResult,
) -> tuple[str, str, str, str | None]:
    if evaluation.asha_action == "prune":
        reason = "; ".join(evaluation.asha_reasons) or "ASHA prune"
        return "pruned", "pruned", "当前 trial 已淘汰", reason
    if evaluation.asha_action == "promote":
        return "promoted", "promoted", "当前 trial 已晋升", None
    if evaluation.passed:
        return "passed", "reached", "当前参数已通过评估", None
    if evaluation.asha_action == "continue":
        reason = "; ".join(evaluation.asha_reasons)
        return "training", "continue", reason or "当前参数需要继续训练", None
    return "failed", "stopped", "当前参数未通过评估", None


def _report_stage(
    reporter: TrainingReporter,
    stage_id: str,
    name: str,
    status: str,
    *,
    processed: int = 0,
    total: int | None = None,
    output_path: Path | None = None,
    warnings: int = 0,
    blocks_training: bool = False,
    error_reason: str | None = None,
    score: float | None = None,
    threshold: float | None = None,
) -> None:
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id=stage_id,
            name=name,
            status=status,
            processed=processed,
            total=total,
            output_path=str(output_path) if output_path is not None else None,
            warning_count=warnings,
            blocks_training=blocks_training,
            error_reason=error_reason,
            score=score,
            threshold=threshold,
        )
    )


def _report_resource(reporter: TrainingReporter) -> None:
    try:
        reporter.report_resource(collect_resource_state())
    except Exception:
        reporter.report_resource(ResourceState())


def _category_scores_from_report(report_path: Path) -> dict[str, float]:
    if not report_path.exists():
        return {}
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    samples = raw.get("samples") or ()
    groups: dict[str, list[float]] = {}
    for sample in samples:
        key = str(sample.get("subproject") or sample.get("primary_error") or "overall")
        try:
            score = float(sample.get("quality_score", 0.0))
        except (TypeError, ValueError):
            continue
        groups.setdefault(key, []).append(score)
    return {
        key: sum(values) / len(values)
        for key, values in groups.items()
        if values
    }


__all__ = [
    "DataInputReport",
    "FullTrainingEvaluationResult",
    "FullTrainingRunConfig",
    "FullTrainingRunResult",
    "TRAINING_STAGES",
    "TrainingStage",
    "run_full_training_pipeline",
    "run_pipeline",
]
