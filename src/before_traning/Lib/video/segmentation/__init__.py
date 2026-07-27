"""公开由谱面驱动的已对齐训练视频分段接口。"""

from before_traning.Lib.video.segmentation.segmentation import (
    SegmentPlanCollection,
    plan_video_segments,
)


__all__ = ["SegmentPlanCollection", "plan_video_segments"]
