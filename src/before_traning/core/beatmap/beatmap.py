from before_traning.conf import Settings
from before_traning.core.beatmap.difficulty import (
    BeatmapDifficultyProcessor,
    DifficultyEntry,
    export_difficulty,
)
from before_traning.core.beatmap.importer import (
    BeatmapImportProcessor,
    import_beatmaps,
)
from before_traning.core.beatmap.pipeline import (
    BEATMAP_TASK_KEYS,
    prepare_beatmaps,
)
from before_traning.core.beatmap.verify import (
    BeatmapVerifyExporter,
    export_verify,
)


def run_beatmap(settings: Settings) -> dict[str, bool]:
    return prepare_beatmaps(settings)


__all__ = [
    "BEATMAP_TASK_KEYS",
    "BeatmapDifficultyProcessor",
    "BeatmapImportProcessor",
    "BeatmapVerifyExporter",
    "DifficultyEntry",
    "export_difficulty",
    "export_verify",
    "import_beatmaps",
    "prepare_beatmaps",
    "run_beatmap",
]
