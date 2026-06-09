"""Assembly entry for beatmap import.

Use:
    from Traning.Lib.beatmap.importing import BeatmapImportProcessor
"""

from Traning.Lib.beatmap.importing.entry import OsuEntry
from Traning.Lib.beatmap.importing.importing import (
    BeatmapImportProcessor,
    OsuOszProcessor,
    build_beatmap_import_processor_from_config_or_default,
    build_osu_osz_processor_from_config_or_default,
)


__all__ = [
    "BeatmapImportProcessor",
    "OsuEntry",
    "OsuOszProcessor",
    "build_beatmap_import_processor_from_config_or_default",
    "build_osu_osz_processor_from_config_or_default",
]
