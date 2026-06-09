from __future__ import annotations

from Traning.conf import ensure_prefect_home

ensure_prefect_home()

from prefect import task

from Traning.conf import Settings
from Traning.core.video.av import av_correspondence


@task(name="av_correspondence", retries=0)
def av_correspondence_task(settings: Settings) -> bool:
    return av_correspondence(settings)
