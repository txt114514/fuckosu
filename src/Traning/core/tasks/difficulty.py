from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PREFECT_HOME", str(Path(__file__).resolve().parents[3] / ".prefect"))

from prefect import task

from Traning.conf import Settings
from Traning.core.beatmap.difficulty import export_difficulty


@task(name="difficulty_export", retries=0)
def export_difficulty_task(settings: Settings) -> bool:
    return export_difficulty(settings)
