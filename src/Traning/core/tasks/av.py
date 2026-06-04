from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PREFECT_HOME", str(Path(__file__).resolve().parents[3] / ".prefect"))

from prefect import task

from Traning.conf import Settings
from Traning.core.video.av import av_correspondence


@task(name="av_correspondence", retries=0)
def av_correspondence_task(settings: Settings) -> bool:
    return av_correspondence(settings)
