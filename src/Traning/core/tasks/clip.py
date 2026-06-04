from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PREFECT_HOME", str(Path(__file__).resolve().parents[3] / ".prefect"))

from prefect import task

from Traning.conf import Settings
from Traning.core.video.clip import crop_video


@task(name="clip", retries=0)
def crop_video_task(settings: Settings) -> bool:
    return crop_video(settings)
