"""Assembly entry for video/audio correspondence.

Use:
    from Traning.Lib.video.av_processing import VideoAVProcessor
"""

from Traning.Lib.video.av_processing.av_processing import (
    AVCorrespondenceProcessor,
    VideoAVProcessor,
    build_av_correspondence_processor_from_config_or_default,
)


__all__ = [
    "AVCorrespondenceProcessor",
    "VideoAVProcessor",
    "build_av_correspondence_processor_from_config_or_default",
]
