from __future__ import annotations

from Traning.conf import ensure_prefect_home

ensure_prefect_home()

from prefect import task

from Traning.conf import Settings
from Traning.core.beatmap.importer import import_beatmaps


@task(name="import_beatmaps", retries=0)
def import_beatmaps_task(settings: Settings) -> bool:
    return import_beatmaps(settings)
