from __future__ import annotations

from dataclasses import dataclass

from before_traning.Lib.beatmap.standard import ParsedStandardBeatmap
from before_traning.Lib.video.segmentation.planner import (
    SegmentPlan,
    build_long_sequence_plans,
    build_segment_plans,
)


@dataclass(frozen=True)
class SegmentPlanCollection:
    atomic: tuple[SegmentPlan, ...]
    long_sequence: tuple[SegmentPlan, ...]

    @property
    def all(self) -> tuple[SegmentPlan, ...]:
        return (*self.atomic, *self.long_sequence)


def plan_video_segments(
    beatmap: ParsedStandardBeatmap,
    *,
    video_duration_seconds: float,
    approach_preempt_ratio: float,
    pre_context_jitter_seconds: float,
    post_context_seconds: float,
    min_circle_overlap_ratio: float,
    priority_merge_window_ms: int,
    use_priority_merge: bool,
    build_long_sequences: bool,
    long_sequence_max_objects: int,
    long_sequence_max_duration_seconds: float,
) -> SegmentPlanCollection:
    atomic = build_segment_plans(
        list(beatmap.hit_objects),
        approach_preempt_ratio=approach_preempt_ratio,
        circle_size=beatmap.circle_size,
        min_circle_overlap_ratio=min_circle_overlap_ratio,
        priority_merge_window_ms=priority_merge_window_ms,
        use_priority_merge=use_priority_merge,
        approach_preempt_seconds=beatmap.approach_preempt_ms / 1000.0,
        pre_context_jitter_seconds=pre_context_jitter_seconds,
        post_context_seconds=post_context_seconds,
        video_duration_seconds=video_duration_seconds,
    )
    long_sequence = (
        build_long_sequence_plans(
            atomic,
            approach_preempt_seconds=(
                beatmap.approach_preempt_ms / 1000.0
            ),
            approach_preempt_ratio=approach_preempt_ratio,
            pre_context_jitter_seconds=pre_context_jitter_seconds,
            post_context_seconds=post_context_seconds,
            video_duration_seconds=video_duration_seconds,
            max_objects=long_sequence_max_objects,
            max_duration_seconds=long_sequence_max_duration_seconds,
        )
        if build_long_sequences
        else []
    )
    return SegmentPlanCollection(
        atomic=tuple(atomic),
        long_sequence=tuple(long_sequence),
    )


__all__ = ["SegmentPlanCollection", "plan_video_segments"]
