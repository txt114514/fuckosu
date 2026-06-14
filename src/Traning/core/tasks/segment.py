from __future__ import annotations

from Traning.conf import ensure_prefect_home

ensure_prefect_home()

from prefect import task

from Traning.conf import Settings
from Traning.core.tasks import require_success
from Traning.core.video.segment import segment_videos


@task(name="video_segment", retries=0)
def segment_videos_task(settings: Settings) -> bool:
    return require_success("video_segment", segment_videos(settings))
