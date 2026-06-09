from __future__ import annotations

from Traning.conf import ensure_prefect_home

ensure_prefect_home()

from prefect import task

from Traning.conf import Settings
from Traning.core.beatmap.difficulty import export_difficulty


@task(name="difficulty_export", retries=0)
def export_difficulty_task(settings: Settings) -> bool:
    return export_difficulty(settings)
