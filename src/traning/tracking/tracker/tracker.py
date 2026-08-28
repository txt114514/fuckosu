"""确定性多目标轨迹生命周期状态机。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from traning.config import TrackingConfig
from traning.contracts import (
    AssociationStatus,
    CandidateObservation,
    TrackedObservation,
    TrackLifecycle,
)
from traning.tracking.association import (
    AssociationResult,
    TrackAssociationView,
    associate_candidates,
)


@dataclass(slots=True)
class _TrackState:
    track_id: str
    lifecycle: TrackLifecycle
    age: int
    missed_frames: int
    last_seen_ms: float
    last_candidate: CandidateObservation
    observation: TrackedObservation


class MultiObjectTracker:
    """只使用当前及历史候选更新轨迹，不读取真值或未来帧。"""

    def __init__(self, config: TrackingConfig) -> None:
        if not isinstance(config, TrackingConfig):
            raise TypeError("config 必须是 TrackingConfig")
        self._config = config
        self.reset()

    def update(
        self,
        candidates: Iterable[CandidateObservation],
        *,
        frame_id: str | None = None,
        frame_index: int | None = None,
        timestamp_ms: float | None = None,
    ) -> tuple[TrackedObservation, ...]:
        """关联一个已处理帧并推进轨迹。

        ``missed_frames`` 统计成功处理但未匹配的帧次数，而不是源视频
        ``frame_index`` 的数值差；因此抽帧输入可以保留原始帧编号，同时
        生命周期仍严格按 tracker 实际看到的帧推进。
        """

        candidate_tuple = tuple(candidates)
        candidate_by_id = _index_candidates(candidate_tuple)
        resolved_frame_id, resolved_frame_index, resolved_timestamp_ms = (
            _resolve_frame_context(
                candidate_tuple,
                frame_id=frame_id,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        )
        self._validate_monotonic_frame(resolved_frame_index, resolved_timestamp_ms)
        track_views = tuple(
            TrackAssociationView(
                track_id=state.track_id,
                last_candidate=state.last_candidate,
                missed_frames=state.missed_frames,
            )
            for state in self._ordered_states()
        )
        association = associate_candidates(track_views, candidate_tuple, self._config)
        _validate_association_result(
            association,
            track_ids=frozenset(self._tracks),
            candidate_ids=frozenset(candidate_by_id),
        )

        output: list[TrackedObservation] = []
        for match in sorted(association.matches, key=lambda item: item.track_id):
            state = self._tracks[match.track_id]
            candidate = candidate_by_id[match.candidate_id]
            observation = TrackedObservation(
                track_id=state.track_id,
                frame_id=resolved_frame_id,
                frame_index=resolved_frame_index,
                timestamp_ms=resolved_timestamp_ms,
                lifecycle=TrackLifecycle.ACTIVE,
                association=AssociationStatus.MATCHED,
                association_confidence=match.confidence,
                track_age=state.age + 1,
                missed_frames=0,
                time_since_seen_ms=0.0,
                candidate=candidate,
                association_cost=match.cost.total,
            )
            state.lifecycle = TrackLifecycle.ACTIVE
            state.age += 1
            state.missed_frames = 0
            state.last_seen_ms = resolved_timestamp_ms
            state.last_candidate = candidate
            state.observation = observation
            output.append(observation)

        expired_track_ids: list[str] = []
        for track_id in sorted(association.unmatched_track_ids):
            state = self._tracks[track_id]
            missed_frames = state.missed_frames + 1
            lifecycle = (
                TrackLifecycle.EXPIRED
                if missed_frames > self._config.max_missed_frames
                else TrackLifecycle.MISSING
            )
            observation = TrackedObservation(
                track_id=track_id,
                frame_id=resolved_frame_id,
                frame_index=resolved_frame_index,
                timestamp_ms=resolved_timestamp_ms,
                lifecycle=lifecycle,
                association=AssociationStatus.UNMATCHED,
                association_confidence=0.0,
                track_age=state.age + 1,
                missed_frames=missed_frames,
                time_since_seen_ms=resolved_timestamp_ms - state.last_seen_ms,
                candidate=None,
            )
            state.lifecycle = lifecycle
            state.age += 1
            state.missed_frames = missed_frames
            state.observation = observation
            output.append(observation)
            if lifecycle is TrackLifecycle.EXPIRED:
                expired_track_ids.append(track_id)

        for candidate_id in sorted(association.unmatched_candidate_ids):
            candidate = candidate_by_id[candidate_id]
            state = self._create_track(candidate, resolved_timestamp_ms)
            output.append(state.observation)

        for track_id in expired_track_ids:
            del self._tracks[track_id]

        self._last_frame_index = resolved_frame_index
        self._last_timestamp_ms = resolved_timestamp_ms
        output.sort(key=lambda observation: observation.track_id)
        return tuple(output)

    process = update

    def snapshot(self) -> tuple[TrackedObservation, ...]:
        """返回当前未过期轨迹的不可变有序观测快照。"""

        return tuple(state.observation for state in self._ordered_states())

    def reset(self) -> None:
        """清空状态并开始新的、从 1 递增的轨迹 ID 空间。"""

        self._tracks: dict[str, _TrackState] = {}
        self._next_track_number = 1
        self._last_frame_index: int | None = None
        self._last_timestamp_ms: float | None = None

    def _validate_monotonic_frame(self, frame_index: int, timestamp_ms: float) -> None:
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index 必须严格递增")
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        ):
            raise ValueError("timestamp_ms 必须单调不减")

    def _ordered_states(self) -> tuple[_TrackState, ...]:
        return tuple(self._tracks[key] for key in sorted(self._tracks))

    def _create_track(
        self, candidate: CandidateObservation, timestamp_ms: float
    ) -> _TrackState:
        track_id = f"track-{self._next_track_number:08d}"
        self._next_track_number += 1
        observation = TrackedObservation(
            track_id=track_id,
            frame_id=candidate.frame_id,
            frame_index=candidate.frame_index,
            timestamp_ms=candidate.timestamp_ms,
            lifecycle=TrackLifecycle.NEW,
            association=AssociationStatus.CREATED,
            association_confidence=candidate.confidence,
            track_age=1,
            missed_frames=0,
            time_since_seen_ms=0.0,
            candidate=candidate,
        )
        state = _TrackState(
            track_id=track_id,
            lifecycle=TrackLifecycle.NEW,
            age=1,
            missed_frames=0,
            last_seen_ms=timestamp_ms,
            last_candidate=candidate,
            observation=observation,
        )
        self._tracks[track_id] = state
        return state


def _resolve_frame_context(
    candidates: tuple[CandidateObservation, ...],
    *,
    frame_id: str | None,
    frame_index: int | None,
    timestamp_ms: float | None,
) -> tuple[str, int, float]:
    if candidates:
        candidate_frame_indices = {candidate.frame_index for candidate in candidates}
        candidate_timestamps = {candidate.timestamp_ms for candidate in candidates}
        candidate_frame_ids = {candidate.frame_id for candidate in candidates}
        if len(candidate_frame_indices) != 1 or len(candidate_timestamps) != 1:
            raise ValueError(
                "同一 update 的候选必须属于相同 frame_index 和 timestamp_ms"
            )
        if len(candidate_frame_ids) != 1:
            raise ValueError("同一 update 的候选必须属于相同 frame_id")
        candidate_frame_id = candidates[0].frame_id
        candidate_frame_index = candidates[0].frame_index
        candidate_timestamp_ms = candidates[0].timestamp_ms
        if frame_id is None:
            frame_id = candidate_frame_id
        elif frame_id != candidate_frame_id:
            raise ValueError("frame_id 与候选帧不一致")
        if frame_index is None:
            frame_index = candidate_frame_index
        elif frame_index != candidate_frame_index:
            raise ValueError("frame_index 与候选帧不一致")
        if timestamp_ms is None:
            timestamp_ms = candidate_timestamp_ms
        elif timestamp_ms != candidate_timestamp_ms:
            raise ValueError("timestamp_ms 与候选帧不一致")
    elif frame_id is None or frame_index is None or timestamp_ms is None:
        raise ValueError("空候选帧必须显式提供 frame_id、frame_index 和 timestamp_ms")

    if not isinstance(frame_id, str):
        raise TypeError("frame_id 必须是字符串")
    if not frame_id or frame_id != frame_id.strip():
        raise ValueError("frame_id 必须是非空且无首尾空格的标识符")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise TypeError("frame_index 必须是整数")
    if frame_index < 0:
        raise ValueError("frame_index 不得为负数")
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, (int, float)):
        raise TypeError("timestamp_ms 必须是数值")
    resolved_timestamp = float(timestamp_ms)
    if not math.isfinite(resolved_timestamp) or resolved_timestamp < 0.0:
        raise ValueError("timestamp_ms 必须是有限非负数")
    return frame_id, frame_index, resolved_timestamp


def _index_candidates(
    candidates: tuple[CandidateObservation, ...],
) -> dict[str, CandidateObservation]:
    result: dict[str, CandidateObservation] = {}
    for candidate in candidates:
        if not isinstance(candidate, CandidateObservation):
            raise TypeError("candidates 只能包含 CandidateObservation")
        if candidate.candidate_id in result:
            raise ValueError(f"candidate_id 重复：{candidate.candidate_id}")
        result[candidate.candidate_id] = candidate
    return result


def _validate_association_result(
    result: AssociationResult,
    *,
    track_ids: frozenset[str],
    candidate_ids: frozenset[str],
) -> None:
    matched_track_ids = [match.track_id for match in result.matches]
    matched_candidate_ids = [match.candidate_id for match in result.matches]
    if len(matched_track_ids) != len(set(matched_track_ids)):
        raise ValueError("association.matches 含有重复 track_id")
    if len(matched_candidate_ids) != len(set(matched_candidate_ids)):
        raise ValueError("association.matches 含有重复 candidate_id")
    result_track_ids = frozenset(matched_track_ids) | frozenset(
        result.unmatched_track_ids
    )
    result_candidate_ids = frozenset(matched_candidate_ids) | frozenset(
        result.unmatched_candidate_ids
    )
    if result_track_ids != track_ids:
        raise ValueError("association result 未精确覆盖输入 tracks")
    if result_candidate_ids != candidate_ids:
        raise ValueError("association result 未精确覆盖输入 candidates")
    if frozenset(matched_track_ids) & frozenset(result.unmatched_track_ids):
        raise ValueError("matched track 不得同时出现在 unmatched_track_ids")
    if frozenset(matched_candidate_ids) & frozenset(result.unmatched_candidate_ids):
        raise ValueError("matched candidate 不得同时出现在 unmatched_candidate_ids")


__all__ = ("MultiObjectTracker",)
