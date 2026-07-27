"""定义批次最佳结果图集所需的帧、trial 与请求契约。"""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from traning.state.experiment_schema import TrialParameters


EVALUATION_SUBPROJECTS = (
    "single_point",
    "slider",
    "multi_point",
    "point_slider",
    "spinner",
    "long_sequence",
)
ErrorDomain = Literal["none", "spatial", "temporal", "decision"]


class FrameEvaluation(BaseModel):
    """保存一帧的通过状态、错误归因和显式 osu/video 预测坐标。"""

    sample_key: str
    frame_index: int
    passed: bool
    target_source_index: int | None = None
    predicted_osu_xy: tuple[float, float] | None = None
    predicted_video_xy: tuple[float, float] | None = None
    action: str | None = None
    action_probability: float | None = None
    primary_error: ErrorDomain = "none"
    error_tags: tuple[str, ...] = ()
    failure_reason: str | None = None
    spatial_error: float | None = None
    temporal_error_ms: float | None = None
    frequency_limited: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("frame_index")
    @classmethod
    def _nonnegative_frame_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("frame_index must be nonnegative")
        return value

    @field_validator("predicted_osu_xy", "predicted_video_xy")
    @classmethod
    def _finite_optional_point(
        cls,
        value: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if value is not None and any(not isfinite(item) for item in value):
            raise ValueError("predicted points must be finite")
        return value

    @field_validator("spatial_error", "temporal_error_ms")
    @classmethod
    def _finite_optional_metric(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("error metrics must be finite")
        return value

    @field_validator("action_probability")
    @classmethod
    def _optional_probability(cls, value: float | None) -> float | None:
        if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
            raise ValueError("action_probability must be finite and in 0..1")
        return value


class TrialGalleryEvaluation(BaseModel):
    trial_id: str
    score: float
    score_version: str = "external"
    parameters: TrialParameters
    frames: tuple[FrameEvaluation, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def _finite_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("score must be finite")
        return value


class BatchGalleryRequest(BaseModel):
    """聚合同一批次可比较的 trial，并提供确定性的最佳 trial 选择。"""

    batch_id: str
    trials: tuple[TrialGalleryEvaluation, ...]
    random_seed: int = 2026
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("trials")
    @classmethod
    def _require_trials(
        cls,
        value: tuple[TrialGalleryEvaluation, ...],
    ) -> tuple[TrialGalleryEvaluation, ...]:
        if not value:
            raise ValueError("trials must not be empty")
        return value

    @model_validator(mode="after")
    def _require_one_score_version(self) -> BatchGalleryRequest:
        # 不同评分语义的数值不可直接排序，必须先在上游完成版本迁移。
        versions = {trial.score_version for trial in self.trials}
        if len(versions) != 1:
            raise ValueError("all trials in one batch must use the same score_version")
        return self

    @property
    def best_trial(self) -> TrialGalleryEvaluation:
        return min(self.trials, key=lambda trial: (-trial.score, trial.trial_id))


def load_batch_gallery_request(path: Path) -> BatchGalleryRequest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read gallery request: {path}") from error
    return BatchGalleryRequest.model_validate(raw)


__all__ = [
    "BatchGalleryRequest",
    "EVALUATION_SUBPROJECTS",
    "ErrorDomain",
    "FrameEvaluation",
    "TrialGalleryEvaluation",
    "load_batch_gallery_request",
]
