"""逐 track 隔离且批次提交原子的 Temporal Belief runtime。"""

from __future__ import annotations

from collections.abc import Sequence

from traning.state import BeliefState, TrackLifecycle, TrackedObservation

from .encoder import PerTrackBeliefEncoder


class PerTrackBeliefRuntime:
    """维护显式公共 belief，不保存模型私有 hidden side state。"""

    def __init__(self, encoder: PerTrackBeliefEncoder) -> None:
        if not isinstance(encoder, PerTrackBeliefEncoder):
            raise TypeError("encoder 必须是 PerTrackBeliefEncoder")
        self.encoder = encoder
        self._states: dict[str, BeliefState] = {}
        self._last_frame_index: int | None = None
        self._last_timestamp_ms: float | None = None

    def step(
        self,
        observations: Sequence[TrackedObservation],
    ) -> tuple[BeliefState, ...]:
        """按稳定 ID 推进整批观测，全部成功后才提交状态。"""

        checked = self._checked_observations(observations)
        incoming_track_ids = {item.track_id for item in checked}
        omitted_track_ids = set(self._states).difference(incoming_track_ids)
        if omitted_track_ids:
            rendered = ", ".join(sorted(omitted_track_ids))
            raise ValueError(f"runtime 批次遗漏现存轨迹：{rendered}")
        if not checked:
            return ()
        frame_index = checked[0].frame_index
        timestamp_ms = checked[0].timestamp_ms
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("runtime frame_index 必须严格递增")
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        ):
            raise ValueError("runtime timestamp_ms 不得回退")
        staged = dict(self._states)
        results: list[BeliefState] = []
        for observation in checked:
            belief = self.encoder.step(observation, staged.get(observation.track_id))
            results.append(belief)
            if observation.lifecycle is TrackLifecycle.EXPIRED:
                staged.pop(observation.track_id, None)
            else:
                staged[observation.track_id] = belief
        # 上面任一步异常都不会运行到此赋值，保证整批 state 原子性。
        self._states = staged
        self._last_frame_index = frame_index
        self._last_timestamp_ms = timestamp_ms
        return tuple(results)

    def state_for(self, track_id: str) -> BeliefState | None:
        """读取单条轨迹的不可变 belief。"""

        if not isinstance(track_id, str):
            raise TypeError("track_id 必须是字符串")
        if not track_id or track_id != track_id.strip():
            raise ValueError("track_id 不得为空且不得有首尾空格")
        return self._states.get(track_id)

    def snapshot(self) -> tuple[BeliefState, ...]:
        """返回按稳定 track ID 排序的不可变状态快照。"""

        return tuple(self._states[key] for key in sorted(self._states))

    def clear(self) -> None:
        """明确清空所有运行时轨迹状态。"""

        self._states = {}
        self._last_frame_index = None
        self._last_timestamp_ms = None

    @staticmethod
    def _checked_observations(
        observations: Sequence[TrackedObservation],
    ) -> tuple[TrackedObservation, ...]:
        if isinstance(observations, (str, bytes)) or not isinstance(
            observations, Sequence
        ):
            raise TypeError("observations 必须是 TrackedObservation 序列")
        checked = tuple(observations)
        if any(not isinstance(item, TrackedObservation) for item in checked):
            raise TypeError("observations 只能包含 TrackedObservation")
        track_ids = tuple(item.track_id for item in checked)
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("单个 runtime 批次不得重复 track_id")
        if checked:
            frame_identity = (
                checked[0].frame_id,
                checked[0].frame_index,
                checked[0].timestamp_ms,
            )
            if any(
                (item.frame_id, item.frame_index, item.timestamp_ms) != frame_identity
                for item in checked[1:]
            ):
                raise ValueError("同一 runtime 批次必须来自同一帧")
        return tuple(sorted(checked, key=lambda item: item.track_id))


__all__ = ("PerTrackBeliefRuntime",)
