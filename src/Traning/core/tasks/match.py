from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PREFECT_HOME", str(Path(__file__).resolve().parents[3] / ".prefect"))

from prefect import task

from Traning.conf import Settings
from Traning.core.video.match import match_videos


@task(name="video_match", retries=0)
def match_videos_task(settings: Settings) -> bool:
    return match_videos(settings)
