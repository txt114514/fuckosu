from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


FullFlowStageStatus = Literal[
    "PENDING",
    "READY",
    "RUNNING",
    "PASSED",
    "WARNING",
    "FAILED",
    "SKIPPED",
    "INTERRUPTED",
    "RESTORED",
    "COMPLETED",
    "LOCKED",
]


@dataclass(frozen=True)
class FullFlowStageSpec:
    stage_id: str
    display_name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    skippable: bool = False
    blocking: bool = True
    resumable: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


FULL_FLOW_STAGES: tuple[FullFlowStageSpec, ...] = (
    FullFlowStageSpec(
        "SOURCE_CHANGE_CHECK",
        "原始数据变更检测",
        inputs=("before_config", "matched_manifest"),
        outputs=("raw_data_report",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "BEFORE_TRAINING",
        "训练前处理",
        inputs=("before_config",),
        outputs=("processed_before_traning_outputs",),
        dependencies=("SOURCE_CHANGE_CHECK",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "DATASET_CONVERSION",
        "原始数据到训练集转换",
        inputs=("before_traning_outputs",),
        outputs=("training_dataset",),
        dependencies=("BEFORE_TRAINING",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "SPLIT_VALIDATION",
        "split 构建或校验",
        inputs=("training_dataset", "training_config"),
        outputs=("split_manifest",),
        dependencies=("DATASET_CONVERSION",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "DATA_QUALITY_CHECK",
        "schema 与数据质量检查",
        inputs=("training_config", "split_manifest"),
        outputs=("data_input_report",),
        dependencies=("SPLIT_VALIDATION",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "ENVIRONMENT_PREFLIGHT",
        "环境、CUDA、磁盘和配置预检",
        inputs=("training_config", "device"),
        outputs=("startup_report",),
        dependencies=("DATA_QUALITY_CHECK",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "RESUME_DISCOVERY",
        "继承状态发现与兼容性检查",
        inputs=("inherit_from", "resume_policy"),
        outputs=("resume_report",),
        dependencies=("ENVIRONMENT_PREFLIGHT",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "RESUME_RESTORE",
        "真实 checkpoint 恢复",
        inputs=("stage_checkpoints",),
        outputs=("restored_training_state",),
        dependencies=("RESUME_DISCOVERY",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "RAMP_TRAINING",
        "受控渐进放大",
        inputs=("training_config", "resume_report"),
        outputs=("ramp_manifest", "level_outputs"),
        dependencies=("RESUME_RESTORE",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "FINAL_READINESS",
        "最终 readiness",
        inputs=("ramp_manifest",),
        outputs=("final_readiness",),
        dependencies=("RAMP_TRAINING",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "FULL_TRAINING",
        "全模型正式训练",
        inputs=("target_config", "resume_report"),
        outputs=("full_training_run",),
        dependencies=("FINAL_READINESS",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "FINAL_EVALUATION",
        "最终评估",
        inputs=("full_training_run", "score_report"),
        outputs=("final_score",),
        dependencies=("FULL_TRAINING",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "MODEL_EXPORT",
        "模型导出",
        inputs=("checkpoints",),
        outputs=("model_artifact",),
        dependencies=("FINAL_EVALUATION",),
        skippable=True,
        blocking=False,
    ),
    FullFlowStageSpec(
        "INHERITANCE_FINALIZATION",
        "inheritance 继承包",
        inputs=("checkpoints", "score_report"),
        outputs=("inheritance_manifest",),
        dependencies=("MODEL_EXPORT",),
        skippable=False,
    ),
    FullFlowStageSpec(
        "REPORT_GENERATION",
        "运行报告",
        inputs=("stage_states",),
        outputs=("full_flow_report",),
        dependencies=("INHERITANCE_FINALIZATION",),
        skippable=False,
    ),
)


STAGE_BY_ID = {stage.stage_id: stage for stage in FULL_FLOW_STAGES}
CRITICAL_STAGE_IDS = frozenset(
    stage.stage_id for stage in FULL_FLOW_STAGES if not stage.skippable
)


def stage_ids() -> tuple[str, ...]:
    return tuple(stage.stage_id for stage in FULL_FLOW_STAGES)


def validate_stage_id(stage_id: str) -> str:
    normalized = stage_id.strip().upper().replace("-", "_")
    if normalized not in STAGE_BY_ID:
        raise ValueError(f"unknown full-flow stage: {stage_id}")
    return normalized


__all__ = [
    "CRITICAL_STAGE_IDS",
    "FULL_FLOW_STAGES",
    "FullFlowStageSpec",
    "FullFlowStageStatus",
    "STAGE_BY_ID",
    "stage_ids",
    "validate_stage_id",
]
