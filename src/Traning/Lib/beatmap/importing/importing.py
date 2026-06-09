from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.beatmap.importing.scanner import OszScannerMixin
from Traning.Lib.beatmap.importing.wrapup import OszImportWrapUpMixin
from Traning.Lib.beatmap.importing.writer import OszImportWriterMixin
from Traning.Lib.beatmap.package import PackageUpdater
from Traning.Lib.common.batch import read_config_values
from Traning.Lib.common.pathspec import suffix_spec
from Traning.Lib.defaults import DEFAULT_SETTINGS as DEFAULTS
from Traning.conf import Settings
from Traning.conf.legacy_config import (
    OSU_OSZ_PROCESSOR_CONFIG_SPECS,
    ConfigReader,
    assign_group,
    build_from_config_or_default,
    settings_namespace,
)
from Traning.state.process_status import ProcessStatusManager


def _load_osu_osz_processor_config(config: ConfigReader) -> dict[str, object]:
    return read_config_values(config, OSU_OSZ_PROCESSOR_CONFIG_SPECS)


def build_osu_osz_processor_from_config_or_default(
    config_path: Path | None = None,
) -> "OsuOszProcessor":
    return build_from_config_or_default(
        OsuOszProcessor,
        [_load_osu_osz_processor_config],
        config_path=config_path,
        default_builder=OsuOszProcessor,
    )


def build_beatmap_import_processor_from_config_or_default(
    config_path: Path | None = None,
) -> "BeatmapImportProcessor":
    return build_from_config_or_default(
        BeatmapImportProcessor,
        [_load_osu_osz_processor_config],
        config_path=config_path,
        default_builder=BeatmapImportProcessor,
    )


class OsuOszProcessor(OszImportWrapUpMixin, OszImportWriterMixin, OszScannerMixin):
    def __init__(
        self,
        settings: Settings = DEFAULTS,
        **overrides: object,
    ):
        if not isinstance(settings, Settings):
            overrides = {"export_dir": settings, **overrides}
            settings = DEFAULTS

        config = settings_namespace(settings, processor="beatmap_import", overrides=overrides)
        assign_group(self, config, "beatmap_import")
        self.export_dir = Path(self.export_dir)
        self.target_root = Path(self.target_root)
        self.keyword = config.keyword.lower()
        self.osz_file_spec = suffix_spec((".osz",))
        self.osu_file_spec = suffix_spec((".osu",))
        self.mp3_file_spec = suffix_spec((".mp3",))
        self.updater = PackageUpdater(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        self.status_manager = ProcessStatusManager(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )

        self.success_count = 0
        self.skip_count = 0
        self.fail_count = 0


class BeatmapImportProcessor(OsuOszProcessor):
    """Task-aligned name for the beatmap import core processor."""


def main():
    processor = build_beatmap_import_processor_from_config_or_default()
    processor.run()


if __name__ == "__main__":
    main()
