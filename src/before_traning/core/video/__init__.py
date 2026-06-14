from before_traning.conf import Settings
from before_traning.core.video.av import av_correspondence
from before_traning.core.video.clip import crop_video
from before_traning.core.video.match import match_videos
from before_traning.core.video.segment import segment_videos


def prepare_videos(settings: Settings) -> dict[str, bool]:
    from before_traning.core.video.pipeline import prepare_videos as prepare

    return prepare(settings)


__all__ = [
    "av_correspondence",
    "crop_video",
    "match_videos",
    "prepare_videos",
    "segment_videos",
]
