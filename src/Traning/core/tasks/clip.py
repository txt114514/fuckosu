from __future__ import annotations

from Traning.conf import ensure_prefect_home

ensure_prefect_home()

from prefect import task

from Traning.conf import Settings
from Traning.core.video.clip import crop_video
from Traning.core.tasks import require_success


@task(name="clip", retries=0)
def crop_video_task(settings: Settings) -> bool:
    return require_success("clip", crop_video(settings))
