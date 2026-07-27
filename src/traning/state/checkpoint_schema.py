"""定义带 lineage 和可恢复组件声明的训练检查点元数据。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator

from traning.state.experiment_schema import CurriculumStage


class CheckpointMetadata(BaseModel):
    """记录检查点位置、训练进度、父节点及可恢复状态范围。"""

    checkpoint_id: str
    trial_id: str
    curriculum_stage: CurriculumStage
    rung: int
    global_step: int
    path: Path
    parent_checkpoint_id: str | None = None
    includes_optimizer: bool = True
    includes_scheduler: bool = True
    includes_amp_scaler: bool = True

    @field_validator("rung", "global_step")
    @classmethod
    def _nonnegative_integer(cls, value: int) -> int:
        if value < 0:
            raise ValueError("checkpoint counters must be nonnegative")
        return value


__all__ = ["CheckpointMetadata"]
