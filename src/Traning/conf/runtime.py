from __future__ import annotations

import os
from pathlib import Path

from Traning.conf.settings import REPO_ROOT


def ensure_prefect_home(repo_root: Path = REPO_ROOT) -> Path:
    prefect_home = repo_root / ".prefect"
    os.environ.setdefault("PREFECT_HOME", str(prefect_home))
    return prefect_home
