from __future__ import annotations

from Traning.conf import Settings
from Traning.core.beatmap.difficulty import export_difficulty
from Traning.core.beatmap.importer import import_beatmaps
from Traning.core.beatmap.verify import export_verify


def prepare_beatmaps(settings: Settings) -> dict[str, bool]:
    return {
        "import_beatmaps": import_beatmaps(settings),
        "verify_export": export_verify(settings),
        "difficulty_export": export_difficulty(settings),
    }


__all__ = [
    "export_difficulty",
    "export_verify",
    "import_beatmaps",
    "prepare_beatmaps",
]
