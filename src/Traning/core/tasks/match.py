from __future__ import annotations

from Traning.conf import ensure_prefect_home

ensure_prefect_home()

from prefect import task

from Traning.conf import Settings
from Traning.core.video.match import match_videos


@task(name="video_match", retries=0)
def match_videos_task(settings: Settings) -> bool:
    return match_videos(settings)
