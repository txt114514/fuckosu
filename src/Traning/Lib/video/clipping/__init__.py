"""Assembly entry for fixed-region video clipping.

Use:
    from Traning.Lib.video.clipping import VideoClipProcessor
"""

from Traning.Lib.video.clipping.clipping import (
    FixedRegionVideoCropProcessor,
    VideoClipProcessor,
    build_fixed_region_video_crop_processor_from_config_or_default,
)


__all__ = [
    "FixedRegionVideoCropProcessor",
    "VideoClipProcessor",
    "build_fixed_region_video_crop_processor_from_config_or_default",
]
