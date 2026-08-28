"""V2 正式部署路径的有状态单帧编排。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from traning.belief import PerTrackBeliefRuntime
from traning.contracts import (
    BeliefState,
    CandidateObservation,
    DecisionAction,
    DecisionResult,
    OutcomeDistribution,
    RuntimeFrame,
    TrackedObservation,
    TrackLifecycle,
)
from traning.decision import OptimalStoppingPlanner
from traning.data import FrameCoordinateTransform
from traning.outcome import DenseOutcomeModel
from traning.perception import PerceptionRuntime
from traning.tracking import MultiObjectTracker


@dataclass(frozen=True, slots=True)
class RuntimeStepResult:
    """一次完整 runtime step 的不可变、可审计输出。"""

    frame_id: str
    frame_index: int
    timestamp_ms: float
    coordinate_transform_fingerprint: str
    candidates: tuple[CandidateObservation, ...]
    tracks: tuple[TrackedObservation, ...]
    active_beliefs: tuple[BeliefState, ...]
    outcomes: tuple[OutcomeDistribution, ...]
    decision: DecisionResult

    def __post_init__(self) -> None:
        """验证跨层身份、帧上下文和稳定排序没有在编排中漂移。"""

        if not isinstance(self.frame_id, str):
            raise TypeError("frame_id 必须是字符串")
        if not self.frame_id or self.frame_id != self.frame_id.strip():
            raise ValueError("frame_id 必须是非空且无首尾空格的标识符")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index 必须是整数")
        if self.frame_index < 0:
            raise ValueError("frame_index 不得为负数")
        if isinstance(self.timestamp_ms, bool) or not isinstance(
            self.timestamp_ms, (int, float)
        ):
            raise TypeError("timestamp_ms 必须是数值")
        if not math.isfinite(float(self.timestamp_ms)) or self.timestamp_ms < 0.0:
            raise ValueError("timestamp_ms 必须是有限非负数")
        if not isinstance(
            self.coordinate_transform_fingerprint, str
        ) or not self.coordinate_transform_fingerprint.startswith("transform-"):
            raise ValueError("coordinate_transform_fingerprint 必须是有效变换指纹")

        tuple_specs = (
            ("candidates", self.candidates, CandidateObservation),
            ("tracks", self.tracks, TrackedObservation),
            ("active_beliefs", self.active_beliefs, BeliefState),
            ("outcomes", self.outcomes, OutcomeDistribution),
        )
        for field_name, values, item_type in tuple_specs:
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} 必须是 tuple")
            if not all(isinstance(value, item_type) for value in values):
                raise TypeError(f"{field_name} 含有错误类型")
        if not isinstance(self.decision, DecisionResult):
            raise TypeError("decision 必须是 DecisionResult")

        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if candidate_ids != tuple(sorted(candidate_ids)):
            raise ValueError("candidates 必须按 candidate_id 稳定排序")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id 不得重复")
        if not all(
            (
                item.frame_id,
                item.frame_index,
                item.timestamp_ms,
            )
            == (self.frame_id, self.frame_index, float(self.timestamp_ms))
            for item in self.candidates
        ):
            raise ValueError("candidates 必须属于当前帧")

        track_ids = tuple(item.track_id for item in self.tracks)
        if track_ids != tuple(sorted(track_ids)):
            raise ValueError("tracks 必须按 track_id 稳定排序")
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("当前帧 track_id 不得重复")
        if not all(
            (
                item.frame_id,
                item.frame_index,
                item.timestamp_ms,
            )
            == (self.frame_id, self.frame_index, float(self.timestamp_ms))
            for item in self.tracks
        ):
            raise ValueError("tracks 必须属于当前帧")
        tracked_candidate_ids = tuple(
            sorted(
                item.candidate.candidate_id
                for item in self.tracks
                if item.candidate is not None
            )
        )
        if tracked_candidate_ids != candidate_ids:
            raise ValueError("每个 candidate 必须恰好进入一条 tracking 输出")

        belief_ids = tuple(item.track_id for item in self.active_beliefs)
        if belief_ids != tuple(sorted(belief_ids)):
            raise ValueError("active_beliefs 必须按 track_id 稳定排序")
        if len(belief_ids) != len(set(belief_ids)):
            raise ValueError("active belief track_id 不得重复")
        expected_active_ids = tuple(
            item.track_id
            for item in self.tracks
            if item.lifecycle is not TrackLifecycle.EXPIRED
        )
        if belief_ids != expected_active_ids:
            raise ValueError("active_beliefs 必须精确对应当前未过期 tracks")
        if not all(
            item.timestamp_ms == float(self.timestamp_ms)
            for item in self.active_beliefs
        ):
            raise ValueError("active_beliefs 必须属于当前时间")

        outcome_keys = tuple(
            (item.track_id, float(item.horizon_ms)) for item in self.outcomes
        )
        if outcome_keys != tuple(sorted(outcome_keys)):
            raise ValueError("outcomes 必须按 track_id、horizon_ms 稳定排序")
        if len(outcome_keys) != len(set(outcome_keys)):
            raise ValueError("同一 track 与 horizon 的 outcome 不得重复")
        outcome_tracks = {track_id for track_id, _horizon_ms in outcome_keys}
        if outcome_tracks != set(belief_ids):
            raise ValueError("outcomes 必须精确覆盖 active_beliefs")
        for track_id in belief_ids:
            horizons = tuple(
                horizon_ms
                for outcome_track_id, horizon_ms in outcome_keys
                if outcome_track_id == track_id
            )
            if len(horizons) != 2 or horizons[0] != 0.0 or horizons[1] <= 0.0:
                raise ValueError("每条 active belief 必须具有当前与单步等待 outcome")

        if self.decision.action is DecisionAction.CLICK:
            if self.decision.track_id not in set(belief_ids):
                raise ValueError("CLICK 必须绑定当前 active belief")
            if self.decision.outcome not in self.outcomes:
                raise ValueError("CLICK 必须绑定本 step 的当前 outcome")
        elif self.decision.horizon_ms <= 0.0:
            raise ValueError("WAIT 必须具有正等待 horizon")


class V2RuntimePipeline:
    """按唯一正式链路推进感知、跟踪、信念、结果预测和决策。

    ``step`` 会在任何有状态组件运行前完成帧与候选边界校验。若状态边界开始后
    出现异常，pipeline 会锁定并要求调用 ``reset``，从而禁止在组件状态可能不一致
    时继续处理下一帧。
    """

    def __init__(
        self,
        perception_runtime: PerceptionRuntime,
        tracker: MultiObjectTracker,
        belief_runtime: PerTrackBeliefRuntime,
        outcome_model: DenseOutcomeModel,
        planner: OptimalStoppingPlanner,
        coordinate_transform: FrameCoordinateTransform,
    ) -> None:
        if not isinstance(perception_runtime, PerceptionRuntime):
            raise TypeError("perception_runtime 必须是 PerceptionRuntime")
        if not isinstance(tracker, MultiObjectTracker):
            raise TypeError("tracker 必须是 MultiObjectTracker")
        if not isinstance(belief_runtime, PerTrackBeliefRuntime):
            raise TypeError("belief_runtime 必须是 PerTrackBeliefRuntime")
        if not isinstance(outcome_model, DenseOutcomeModel):
            raise TypeError("outcome_model 必须是 DenseOutcomeModel")
        if not isinstance(planner, OptimalStoppingPlanner):
            raise TypeError("planner 必须是 OptimalStoppingPlanner")
        if not isinstance(coordinate_transform, FrameCoordinateTransform):
            raise TypeError("coordinate_transform 必须是 FrameCoordinateTransform")
        if (
            outcome_model.belief_embedding_dim
            != belief_runtime.encoder.flattened_hidden_dim
        ):
            raise ValueError("Outcome 与 Belief 的 embedding 维度必须一致")
        supported_horizons = {
            float(value) for value in outcome_model.config.horizons_ms
        }
        if {0.0, planner.wait_horizon_ms}.difference(supported_horizons):
            raise ValueError("Outcome 配置必须支持 planner 的当前与等待 horizon")

        self._perception_runtime = perception_runtime
        self._tracker = tracker
        self._belief_runtime = belief_runtime
        self._outcome_model = outcome_model
        self._planner = planner
        self._coordinate_transform = coordinate_transform
        self._outcome_model.eval()
        self._last_frame_index: int | None = None
        self._last_timestamp_ms: float | None = None
        self._requires_reset = False

    @property
    def requires_reset(self) -> bool:
        """状态边界异常后是否必须先清空序列状态。"""

        return self._requires_reset

    @property
    def coordinate_transform(self) -> FrameCoordinateTransform:
        """返回 runtime 帧尺寸、训练制品和离线评分共用的坐标身份。"""

        return self._coordinate_transform

    def step(self, frame: RuntimeFrame) -> RuntimeStepResult:
        """处理一帧并仅在完整决策成功后提交外层帧游标。"""

        self._validate_frame(frame)
        raw_candidates = self._perception_runtime.infer(frame)
        candidates = self._checked_candidates(raw_candidates, frame)

        # tracker.update 是第一处有状态边界；之后的异常要求显式 reset。
        state_boundary_started = False
        try:
            state_boundary_started = True
            tracks = self._tracker.update(
                candidates,
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
            )
            tracks = tuple(sorted(tracks, key=lambda item: item.track_id))
            self._belief_runtime.step(tracks)
            # EXPIRED 当帧会被 belief step 返回，但 snapshot 已将它移除。
            active_beliefs = self._belief_runtime.snapshot()
            outcomes = self._predict_outcomes(active_beliefs)
            decision = self._planner.plan(
                active_beliefs,
                outcomes,
                frame.timestamp_ms,
            )
            result = RuntimeStepResult(
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                coordinate_transform_fingerprint=(
                    self._coordinate_transform.transform_fingerprint
                ),
                candidates=candidates,
                tracks=tracks,
                active_beliefs=active_beliefs,
                outcomes=outcomes,
                decision=decision,
            )
        except Exception:
            if state_boundary_started:
                self._requires_reset = True
            raise

        self._last_frame_index = frame.frame_index
        self._last_timestamp_ms = frame.timestamp_ms
        return result

    def reset(self) -> None:
        """清空 tracker、belief 与 pipeline 帧游标，开始一个全新序列。"""

        self._tracker.reset()
        self._belief_runtime.clear()
        self._last_frame_index = None
        self._last_timestamp_ms = None
        self._requires_reset = False

    def _validate_frame(self, frame: RuntimeFrame) -> None:
        if not isinstance(frame, RuntimeFrame):
            raise TypeError("frame 必须是 RuntimeFrame")
        if self._requires_reset:
            raise RuntimeError("上一次 stateful step 失败；继续前必须调用 reset")
        if (
            frame.width != self._coordinate_transform.source_frame_width
            or frame.height != self._coordinate_transform.source_frame_height
        ):
            raise ValueError("RuntimeFrame 尺寸与 runtime 坐标标定尺寸不一致")
        if (
            self._last_frame_index is not None
            and frame.frame_index <= self._last_frame_index
        ):
            raise ValueError("runtime frame_index 必须严格递增")
        if (
            self._last_timestamp_ms is not None
            and frame.timestamp_ms < self._last_timestamp_ms
        ):
            raise ValueError("runtime timestamp_ms 不得回退")

    def _checked_candidates(
        self,
        candidates: tuple[CandidateObservation, ...],
        frame: RuntimeFrame,
    ) -> tuple[CandidateObservation, ...]:
        if not isinstance(candidates, tuple):
            raise TypeError(
                "PerceptionRuntime.infer 必须返回 CandidateObservation tuple"
            )
        if not all(isinstance(item, CandidateObservation) for item in candidates):
            raise TypeError("PerceptionRuntime.infer 只能返回 CandidateObservation")
        candidate_ids = tuple(item.candidate_id for item in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("PerceptionRuntime.infer 返回了重复 candidate_id")
        expected_embedding_dim = self._belief_runtime.encoder.appearance_embedding_dim
        for candidate in candidates:
            if (
                candidate.frame_id,
                candidate.frame_index,
                candidate.timestamp_ms,
            ) != (frame.frame_id, frame.frame_index, frame.timestamp_ms):
                raise ValueError("candidate 帧上下文与 RuntimeFrame 不一致")
            if len(candidate.appearance_embedding) != expected_embedding_dim:
                raise ValueError("candidate embedding 维度与 Belief encoder 不一致")
            if not 0.0 <= candidate.x <= float(frame.width - 1):
                raise ValueError("candidate.x 超出当前帧像素范围")
            if not 0.0 <= candidate.y <= float(frame.height - 1):
                raise ValueError("candidate.y 超出当前帧像素范围")
        return tuple(sorted(candidates, key=lambda item: item.candidate_id))

    def _predict_outcomes(
        self,
        beliefs: tuple[BeliefState, ...],
    ) -> tuple[OutcomeDistribution, ...]:
        horizons = (0.0, self._planner.wait_horizon_ms)
        return tuple(
            self._outcome_model.predict(belief, horizon_ms)
            for belief in beliefs
            for horizon_ms in horizons
        )


__all__ = ("RuntimeStepResult", "V2RuntimePipeline")
