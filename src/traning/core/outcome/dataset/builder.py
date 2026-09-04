"""从逐轨迹 belief 与离线 oracle 状态构造反事实训练样本。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

from traning.state import (
    BeliefState,
    DataSplit,
    DecisionAction,
    OutcomeTrainingSample,
    Point2D,
    require_transform_fingerprint,
)
from traning.core.data import FrameCoordinateTransform
from traning.core.outcome.oracle import (
    HypotheticalClick,
    OracleOutcome,
    OracleState,
    OutcomeOracle,
)


@dataclass(frozen=True, slots=True)
class CounterfactualFrame:
    """同一时刻的 typed beliefs、原帧坐标来源与离线 oracle 状态。"""

    sample_id: str
    split: DataSplit
    source_frame_width: int
    source_frame_height: int
    transform_fingerprint: str
    beliefs: tuple[BeliefState, ...]
    oracle_state: OracleState

    def __post_init__(self) -> None:
        _identifier(self.sample_id, "sample_id")
        if not isinstance(self.split, DataSplit) or self.split is DataSplit.ALL:
            raise ValueError("split 必须是具体 DataSplit")
        for field_name, value in (
            ("source_frame_width", self.source_frame_width),
            ("source_frame_height", self.source_frame_height),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} 必须是整数")
            if value < 1:
                raise ValueError(f"{field_name} 必须为正整数")
        require_transform_fingerprint(self.transform_fingerprint)
        if not isinstance(self.beliefs, tuple) or any(
            not isinstance(belief, BeliefState) for belief in self.beliefs
        ):
            raise TypeError("beliefs 必须是 BeliefState 元组")
        if not isinstance(self.oracle_state, OracleState):
            raise TypeError("oracle_state 必须是 OracleState")
        track_ids = tuple(belief.track_id for belief in self.beliefs)
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("单个 CounterfactualFrame 的 belief.track_id 不得重复")
        if any(
            belief.timestamp_ms != self.oracle_state.timestamp_ms
            for belief in self.beliefs
        ):
            raise ValueError("belief 与 oracle_state 必须属于同一 timestamp_ms")


@dataclass(frozen=True, slots=True)
class CounterfactualOutcomeDataset:
    """单一 split 且只使用一个坐标指纹的反事实 Outcome 样本集合。"""

    split: DataSplit
    records: tuple[OutcomeTrainingSample, ...]
    transform_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.split, DataSplit) or self.split is DataSplit.ALL:
            raise ValueError("split 必须是具体 DataSplit")
        if not isinstance(self.records, tuple):
            raise TypeError("records 必须是 OutcomeTrainingSample 元组")
        if not self.records:
            raise ValueError("records 不得为空")
        require_transform_fingerprint(self.transform_fingerprint)
        if any(
            not isinstance(record, OutcomeTrainingSample) for record in self.records
        ):
            raise TypeError("records 只能包含 OutcomeTrainingSample")
        if any(record.split is not self.split for record in self.records):
            raise ValueError(
                "每条 OutcomeTrainingSample.split 必须与 dataset.split 一致"
            )
        sample_ids = tuple(record.sample_id for record in self.records)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("CounterfactualOutcomeDataset.sample_id 不得重复")


class CounterfactualOutcomeDatasetBuilder:
    """按 frame、track、horizon 稳定次序生成 CLICK 反事实样本。"""

    def __init__(
        self,
        oracle: OutcomeOracle,
        horizons_ms: tuple[float, ...],
        coordinate_transform: FrameCoordinateTransform,
    ) -> None:
        if not isinstance(oracle, OutcomeOracle):
            raise TypeError("oracle 必须是 OutcomeOracle")
        if not isinstance(coordinate_transform, FrameCoordinateTransform):
            raise TypeError("coordinate_transform 必须是 FrameCoordinateTransform")
        if not isinstance(horizons_ms, tuple) or not horizons_ms:
            raise TypeError("horizons_ms 必须是非空元组")
        checked: list[float] = []
        for index, horizon in enumerate(horizons_ms):
            if isinstance(horizon, bool) or not isinstance(horizon, (int, float)):
                raise TypeError(f"horizons_ms[{index}] 必须是数值")
            value = float(horizon)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"horizons_ms[{index}] 必须是有限非负数")
            checked.append(value)
        if any(left >= right for left, right in pairwise(checked)):
            raise ValueError("horizons_ms 必须严格递增")
        self._oracle = oracle
        self._horizons_ms = tuple(checked)
        self._coordinate_transform = coordinate_transform

    @property
    def horizons_ms(self) -> tuple[float, ...]:
        """返回规范化后的严格递增 horizon。"""

        return self._horizons_ms

    def build(
        self, frames: Iterable[CounterfactualFrame]
    ) -> CounterfactualOutcomeDataset:
        """全量校验后，以确定性次序生成 canonical OutcomeTrainingSample。"""

        checked_frames = tuple(frames)
        if not checked_frames:
            raise ValueError("frames 不得为空")
        if any(not isinstance(frame, CounterfactualFrame) for frame in checked_frames):
            raise TypeError("frames 只能包含 CounterfactualFrame")
        sample_ids = tuple(frame.sample_id for frame in checked_frames)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("frames.sample_id 不得重复")
        splits = {frame.split for frame in checked_frames}
        if len(splits) > 1:
            raise ValueError("一次 build 的 frames 必须属于同一 split")
        for frame in checked_frames:
            if (
                frame.source_frame_width
                != self._coordinate_transform.source_frame_width
                or frame.source_frame_height
                != self._coordinate_transform.source_frame_height
            ):
                raise ValueError("CounterfactualFrame 原帧尺寸与坐标标定不一致")
            if (
                frame.transform_fingerprint
                != self._coordinate_transform.transform_fingerprint
            ):
                raise ValueError("CounterfactualFrame 与 builder 的坐标变换指纹不一致")

        records: list[OutcomeTrainingSample] = []
        for frame in sorted(checked_frames, key=lambda item: item.sample_id):
            for belief in sorted(frame.beliefs, key=lambda item: item.track_id):
                # Belief.position_mean 来自 runtime 原帧像素；oracle target 则是
                # canonical osu! 坐标。不允许直接将两者传入同一 scorer。
                bound_prediction = self._coordinate_transform.bind_frame_prediction(
                    x=belief.position_mean.x,
                    y=belief.position_mean.y,
                    source_frame_width=frame.source_frame_width,
                    source_frame_height=frame.source_frame_height,
                )
                canonical_prediction = (
                    self._coordinate_transform.prediction_to_canonical_scoring(
                        bound_prediction
                    )
                )
                for horizon_index, horizon_ms in enumerate(self._horizons_ms):
                    outcome = self._oracle.evaluate(
                        frame.oracle_state,
                        HypotheticalClick(
                            track_id=belief.track_id,
                            horizon_ms=horizon_ms,
                            position=Point2D(
                                canonical_prediction.x,
                                canonical_prediction.y,
                            ),
                        ),
                    )
                    _validate_outcome(outcome, belief.track_id, horizon_ms)
                    records.append(
                        OutcomeTrainingSample(
                            sample_id=_counterfactual_sample_id(
                                frame.sample_id,
                                belief.track_id,
                                horizon_index,
                                horizon_ms,
                            ),
                            split=frame.split,
                            source_sample_id=frame.sample_id,
                            oracle_state_id=frame.oracle_state.state_id,
                            belief=belief,
                            action=DecisionAction.CLICK,
                            action_track_id=belief.track_id,
                            horizon_ms=horizon_ms,
                            target_category=outcome.category,
                            target_score=outcome.score,
                            valid=outcome.valid,
                            expires=outcome.expires,
                            target_object_id=outcome.target_object_id,
                        )
                    )
        split = checked_frames[0].split
        return CounterfactualOutcomeDataset(
            split=split,
            records=tuple(records),
            transform_fingerprint=self._coordinate_transform.transform_fingerprint,
        )


def _counterfactual_sample_id(
    source_sample_id: str,
    track_id: str,
    horizon_index: int,
    horizon_ms: float,
) -> str:
    """用 length-prefix 编码可变标识符，避免分隔符组合产生碰撞。"""

    return (
        f"cf|{len(source_sample_id)}:{source_sample_id}|"
        f"{len(track_id)}:{track_id}|h:{horizon_index:04d}:{horizon_ms.hex()}"
    )


def _validate_outcome(
    outcome: OracleOutcome,
    track_id: str,
    horizon_ms: float,
) -> None:
    if not isinstance(outcome, OracleOutcome):
        raise TypeError("oracle.evaluate 必须返回 OracleOutcome")
    if outcome.track_id != track_id or outcome.horizon_ms != horizon_ms:
        raise ValueError("OracleOutcome 必须对应请求的 track_id 和 horizon_ms")


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} 必须是字符串")
    if not value or value != value.strip():
        raise ValueError(f"{name} 必须非空且无首尾空格")


__all__ = (
    "CounterfactualFrame",
    "CounterfactualOutcomeDataset",
    "CounterfactualOutcomeDatasetBuilder",
)
