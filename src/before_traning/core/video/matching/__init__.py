"""公开录像匹配策略、顺序重命名器及兼容构建入口。"""

from before_traning.core.video.matching.builders import (
    build_video_package_renamer_from_config_or_default,
)
from before_traning.core.video.matching.matching import VideoMatchProcessor
from before_traning.core.video.matching.renamer import VideoMatchRenamer, VideoPackageRenamer


__all__ = [
    "VideoMatchProcessor",
    "VideoMatchRenamer",
    "VideoPackageRenamer",
    "build_video_package_renamer_from_config_or_default",
]
