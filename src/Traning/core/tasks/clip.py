from __future__ import annotations

from Traning.conf import ensure_prefect_home

ensure_prefect_home()

from prefect import task

from Traning.conf import Settings
from Traning.core.video.clip import crop_video


@task(name="clip", retries=0)
def crop_video_task(settings: Settings) -> bool:
    return crop_video(settings)
