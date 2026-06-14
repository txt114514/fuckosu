"""Beatmap-driven segmentation for aligned training videos."""

from before_traning.Lib.video.segmentation.segmentation import (
    SegmentPlanCollection,
    plan_video_segments,
)


__all__ = ["SegmentPlanCollection", "plan_video_segments"]
