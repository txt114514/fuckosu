from __future__ import annotations

from Traning.conf import ensure_prefect_home

ensure_prefect_home()

from prefect import task

from Traning.conf import Settings
from Traning.core.beatmap.verify import export_verify


@task(name="verify_export", retries=0)
def export_verify_task(settings: Settings) -> bool:
    return export_verify(settings)
