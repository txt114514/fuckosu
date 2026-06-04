from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PREFECT_HOME", str(Path(__file__).resolve().parents[3] / ".prefect"))

from prefect import task

from Traning.conf import Settings
from Traning.core.beatmap.verify import export_verify


@task(name="verify_export", retries=0)
def export_verify_task(settings: Settings) -> bool:
    return export_verify(settings)
