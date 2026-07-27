"""公开 AV 对齐处理器及其兼容配置构建入口。"""

from before_traning.core.video.av_processing.av_processing import (
    AVCorrespondenceProcessor,
    VideoAVProcessor,
    build_av_correspondence_processor_from_config_or_default,
)


__all__ = [
    "AVCorrespondenceProcessor",
    "VideoAVProcessor",
    "build_av_correspondence_processor_from_config_or_default",
]
