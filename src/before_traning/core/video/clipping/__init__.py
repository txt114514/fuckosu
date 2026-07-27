"""公开固定区域视频裁剪处理器及其兼容构建入口。"""

from before_traning.core.video.clipping.clipping import (
    FixedRegionVideoCropProcessor,
    VideoClipProcessor,
    build_fixed_region_video_crop_processor_from_config_or_default,
)


__all__ = [
    "FixedRegionVideoCropProcessor",
    "VideoClipProcessor",
    "build_fixed_region_video_crop_processor_from_config_or_default",
]
