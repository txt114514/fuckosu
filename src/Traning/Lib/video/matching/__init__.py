"""Assembly entry for video matching.

Use:
    from Traning.Lib.video.matching import VideoMatchProcessor
"""

from Traning.Lib.video.matching.builders import (
    build_video_package_renamer_from_config_or_default,
)
from Traning.Lib.video.matching.matching import VideoMatchProcessor
from Traning.Lib.video.matching.renamer import VideoMatchRenamer, VideoPackageRenamer


__all__ = [
    "VideoMatchProcessor",
    "VideoMatchRenamer",
    "VideoPackageRenamer",
    "build_video_package_renamer_from_config_or_default",
]
