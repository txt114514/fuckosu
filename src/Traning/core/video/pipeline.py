from __future__ import annotations

from Traning.conf import Settings
from Traning.core.video.av import av_correspondence
from Traning.core.video.clip import crop_video
from Traning.core.video.match import match_videos


def prepare_videos(settings: Settings) -> dict[str, bool]:
    return {
        "video_match": match_videos(settings),
        "av_correspondence": av_correspondence(settings),
        "clip": crop_video(settings),
    }


__all__ = [
    "av_correspondence",
    "crop_video",
    "match_videos",
    "prepare_videos",
]
