"""训练、推理和运行时相互隔离的数据契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor

from package import DataSplit

from .belief import BeliefState
from .common import (
    require_finite,
    require_identifier,
    require_nonnegative,
    require_probability,
    require_transform_fingerprint,
)
from .decision import ActionType
from .observation import Candidate, ObjectType, Point2D
from .outcome import (
    OUTCOME_LOW_SCORE_UPPER,
    OUTCOME_MEDIUM_SCORE_UPPER,
    OutcomeCategory,
)


@dataclass(frozen=True, slots=True)
class GroundTruthObject:
    """仅训练侧可见的 canonical osu! 目标及其类型几何。"""

    object_id: str
    object_type: ObjectType
    position: Point2D
    start_time_ms: float
    end_time_ms: float
    score: float
    radius_osu: float | None = None
    path: tuple[Point2D, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.object_id, "object_id")
        if not isinstance(self.object_type, ObjectType):
            raise TypeError("object_type 必须是 ObjectType")
        if not isinstance(self.position, Point2D):
            raise TypeError("position 必须是 Point2D")
        require_nonnegative(self.start_time_ms, "start_time_ms")
        require_nonnegative(self.end_time_ms, "end_time_ms")
        if self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms 不得早于 start_time_ms")
        require_finite(self.score, "score")
        if not isinstance(self.path, tuple) or any(
            not isinstance(point, Point2D) for point in self.path
        ):
            raise TypeError("path 必须是 Point2D 元组")
        if self.radius_osu is not None:
            require_nonnegative(self.radius_osu, "radius_osu")
            if self.radius_osu == 0.0:
                raise ValueError("radius_osu 必须大于 0")
        if self.object_type is ObjectType.RING:
            if self.radius_osu is None or self.path:
                raise ValueError("RING 真值必须提供正 radius_osu 且不得携带 path")
        elif self.object_type is ObjectType.SLIDER:
            if len(self.path) < 2 or self.position != self.path[0]:
                raise ValueError("SLIDER 真值 path 至少两点且起点必须等于 position")
            if self.radius_osu is not None:
                raise ValueError("SLIDER 真值不得携带 radius_osu")
        elif self.radius_osu is not None or self.path:
            raise ValueError("SPINNER/UNKNOWN 真值不得携带 ring/slider 几何")


@dataclass(frozen=True, slots=True)
class TrainingCandidateRecord:
    """候选观测与训练专属监督的组合。"""

    sample_id: str
    observation: Candidate
    matched_object: GroundTruthObject | None
    is_selected: bool
    temporal_target: tuple[float, ...]

    def __post_init__(self) -> None:
        require_identifier(self.sample_id, "sample_id")
        if not isinstance(self.observation, Candidate):
            raise TypeError("observation 必须是 Candidate")
        if self.matched_object is not None and not isinstance(
            self.matched_object, GroundTruthObject
        ):
            raise TypeError("matched_object 必须是 GroundTruthObject 或 None")
        if not isinstance(self.is_selected, bool):
            raise TypeError("is_selected 必须是 bool")
        if not isinstance(self.temporal_target, tuple):
            raise TypeError("temporal_target 必须是浮点元组")
        for index, value in enumerate(self.temporal_target):
            require_finite(value, f"temporal_target[{index}]")


@dataclass(frozen=True, slots=True)
class InferenceCandidateRecord:
    """推理侧候选记录；类型上不提供任何训练真值。"""

    frame_id: str
    observation: Candidate

    def __post_init__(self) -> None:
        require_identifier(self.frame_id, "frame_id")
        if not isinstance(self.observation, Candidate):
            raise TypeError("observation 必须是 Candidate")
        if self.observation.frame_id != self.frame_id:
            raise ValueError("observation.frame_id 必须与 record.frame_id 一致")


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """运行时输入帧；只携带图像和运行时身份信息。"""

    frame_id: str
    frame_index: int
    timestamp_ms: float
    width: int
    height: int
    image_bytes: bytes

    def __post_init__(self) -> None:
        require_identifier(self.frame_id, "frame_id")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index 必须是整数")
        if self.frame_index < 0:
            raise ValueError("frame_index 不得为负数")
        require_nonnegative(self.timestamp_ms, "timestamp_ms")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.width, self.height)
        ):
            raise TypeError("width 和 height 必须是整数")
        if self.width < 1 or self.height < 1:
            raise ValueError("width 和 height 必须为正整数")
        if not isinstance(self.image_bytes, bytes):
            raise TypeError("image_bytes 必须是 bytes")
        if not self.image_bytes:
            raise ValueError("image_bytes 不得为空")


# 旧名称是 identity alias；VideoFrame 是注册表中的规范名称。
RuntimeFrame = VideoFrame


@dataclass(frozen=True, slots=True)
class FrameBatch:
    """边界校验后的 BCHW 帧张量及其稳定帧身份。"""

    frames: Tensor
    frame_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LabelBatch:
    """按监督头命名的标签张量集合。"""

    labels: Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """训练 step 唯一接收的帧与标签组合契约。"""

    frames: FrameBatch
    labels: LabelBatch


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """完整训练帧及其仅训练侧可见的监督。

    ``transform_fingerprint`` 将原帧尺寸和 osu! -> 原帧标定方程
    绑定到样本，使 target 不能在无感知的情况下换用旧坐标系。
    """

    sample_id: str
    split: DataSplit
    frame_index: int
    timestamp_ms: float
    width: int
    height: int
    image_bytes: bytes
    transform_fingerprint: str
    candidates: tuple[TrainingCandidateRecord, ...]
    ground_truth_objects: tuple[GroundTruthObject, ...]
    selected_candidate_id: str | None

    def __post_init__(self) -> None:
        require_identifier(self.sample_id, "sample_id")
        if not isinstance(self.split, DataSplit) or self.split is DataSplit.ALL:
            raise ValueError("TrainingSample 必须属于具体数据切分")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index 必须是整数")
        if self.frame_index < 0:
            raise ValueError("frame_index 不得为负数")
        require_nonnegative(self.timestamp_ms, "timestamp_ms")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.width, self.height)
        ):
            raise TypeError("width 和 height 必须是整数")
        if not isinstance(self.image_bytes, bytes):
            raise TypeError("image_bytes 必须是 bytes")
        if self.width < 1 or self.height < 1 or not self.image_bytes:
            raise ValueError("训练图像必须具有正尺寸和非空内容")
        require_transform_fingerprint(self.transform_fingerprint)
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(record, TrainingCandidateRecord)
            for record in self.candidates
        ):
            raise TypeError("candidates 必须是 TrainingCandidateRecord 元组")
        if not isinstance(self.ground_truth_objects, tuple) or any(
            not isinstance(target, GroundTruthObject)
            for target in self.ground_truth_objects
        ):
            raise TypeError("ground_truth_objects 必须是 GroundTruthObject 元组")
        if any(record.sample_id != self.sample_id for record in self.candidates):
            raise ValueError("candidate.sample_id 必须与 TrainingSample.sample_id 一致")
        candidate_ids = tuple(
            record.observation.candidate_id for record in self.candidates
        )
        object_ids = tuple(target.object_id for target in self.ground_truth_objects)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("TrainingSample candidate_id 不得重复")
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("TrainingSample object_id 不得重复")
        if self.selected_candidate_id is not None:
            require_identifier(self.selected_candidate_id, "selected_candidate_id")
            if self.selected_candidate_id not in candidate_ids:
                raise ValueError("selected_candidate_id 必须引用本样本候选")
        selected_ids = tuple(
            record.observation.candidate_id
            for record in self.candidates
            if record.is_selected
        )
        expected_selected = (
            () if self.selected_candidate_id is None else (self.selected_candidate_id,)
        )
        if selected_ids != expected_selected:
            raise ValueError("is_selected 标记必须精确对应 selected_candidate_id")


@dataclass(frozen=True, slots=True)
class OutcomeTrainingSample:
    """反事实 Outcome 模型的训练样本；target_score 是归一化 oracle 分数。"""

    sample_id: str
    split: DataSplit
    source_sample_id: str
    oracle_state_id: str
    belief: BeliefState
    action: ActionType
    action_track_id: str | None
    horizon_ms: float
    target_category: OutcomeCategory
    target_score: float
    valid: bool
    expires: bool
    target_object_id: str | None

    def __post_init__(self) -> None:
        require_identifier(self.sample_id, "sample_id")
        if not isinstance(self.split, DataSplit) or self.split is DataSplit.ALL:
            raise ValueError("OutcomeTrainingSample 必须属于具体 DataSplit")
        require_identifier(self.source_sample_id, "source_sample_id")
        require_identifier(self.oracle_state_id, "oracle_state_id")
        if not isinstance(self.belief, BeliefState):
            raise TypeError("belief 必须是 BeliefState")
        if not isinstance(self.action, ActionType):
            raise TypeError("action 必须是 ActionType")
        if self.action is not ActionType.CLICK:
            raise ValueError("Outcome V2 baseline 训练样本只支持 CLICK")
        require_nonnegative(self.horizon_ms, "horizon_ms")
        require_probability(self.target_score, "target_score")
        if not isinstance(self.target_category, OutcomeCategory):
            raise TypeError("target_category 必须是 OutcomeCategory")
        if not isinstance(self.valid, bool) or not isinstance(self.expires, bool):
            raise TypeError("valid 和 expires 必须是布尔值")
        if self.action_track_id is None:
            raise ValueError("CLICK 训练样本必须指定 action_track_id")
        require_identifier(self.action_track_id, "action_track_id")
        if self.action_track_id != self.belief.track_id:
            raise ValueError("action_track_id 必须与 belief.track_id 一致")
        if self.target_object_id is not None:
            require_identifier(self.target_object_id, "target_object_id")

        if self.target_category is OutcomeCategory.INVALID:
            if self.valid or self.target_score != 0.0:
                raise ValueError("INVALID 样本必须 valid=False 且 target_score=0")
        else:
            if not self.valid or self.expires:
                raise ValueError("非 INVALID 样本必须 valid=True 且 expires=False")
            if self.target_object_id is None:
                raise ValueError("有效 Outcome 样本必须记录 target_object_id")
        if self.expires and self.target_category is not OutcomeCategory.INVALID:
            raise ValueError("expires=True 只允许用于 INVALID 样本")
        if (
            self.target_category is OutcomeCategory.LOW
            and self.target_score >= OUTCOME_LOW_SCORE_UPPER
        ):
            raise ValueError("LOW 样本分数必须低于 low threshold")
        if self.target_category is OutcomeCategory.MEDIUM and not (
            OUTCOME_LOW_SCORE_UPPER <= self.target_score < OUTCOME_MEDIUM_SCORE_UPPER
        ):
            raise ValueError("MEDIUM 样本分数必须位于配置区间")
        if (
            self.target_category is OutcomeCategory.HIGH
            and self.target_score < OUTCOME_MEDIUM_SCORE_UPPER
        ):
            raise ValueError("HIGH 样本分数不得低于 medium threshold")
