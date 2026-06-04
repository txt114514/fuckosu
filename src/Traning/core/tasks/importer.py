from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PREFECT_HOME", str(Path(__file__).resolve().parents[3] / ".prefect"))

from prefect import task

from Traning.conf import Settings
from Traning.core.beatmap.importer import import_beatmaps


@task(name="import_beatmaps", retries=0)
def import_beatmaps_task(settings: Settings) -> bool:
    return import_beatmaps(settings)
