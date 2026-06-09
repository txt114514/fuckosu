from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.beatmap.folder_store import BeatmapFolderStore
from Traning.Lib.beatmap.order import OrderFolderWalker
from Traning.Lib.beatmap.verification.parser import VerifyOsuParser
from Traning.Lib.beatmap.verification.steps import VerifyStepsMixin
from Traning.Lib.beatmap.verification.wrapup import VerifyWrapUpMixin
from Traning.Lib.common.batch import FolderBatchProcessor, read_config_values
from Traning.Lib.defaults import DEFAULT_SETTINGS as DEFAULTS
from Traning.conf import Settings
from Traning.conf.legacy_config import (
    VERIFY_EXPORTER_CONFIG_SPECS,
    ConfigReader,
    build_from_config_or_default,
    settings_namespace,
)
from Traning.state.process_status import ProcessStatusManager


def _load_verify_exporter_config(config: ConfigReader) -> dict[str, object]:
    return read_config_values(config, VERIFY_EXPORTER_CONFIG_SPECS)


class VerifyExporter(VerifyWrapUpMixin, VerifyStepsMixin, FolderBatchProcessor):
    def __init__(
        self,
        walker: OrderFolderWalker,
        store: BeatmapFolderStore,
        settings: Settings = DEFAULTS,
        status_manager: ProcessStatusManager | None = None,
        **overrides: object,
    ):
        if not isinstance(settings, Settings):
            overrides = {"verify_filename": settings, **overrides}
            settings = DEFAULTS

        config = settings_namespace(settings, processor="verify", overrides=overrides)
        self.walker = walker
        self.store = store
        self.verify_filename = config.verify_filename
        super().__init__(config.failed_filename)
        self.parser = VerifyOsuParser()
        self.status_manager = status_manager or ProcessStatusManager(
            target_root=str(store.target_root),
            order_filename=store.order_filename,
        )


class BeatmapVerifyExporter(VerifyExporter):
    """Task-aligned name for the beatmap verify export processor."""


def _build_verify_exporter_from_config(
    settings: Settings = DEFAULTS,
    **overrides: object,
) -> BeatmapVerifyExporter:
    config = settings_namespace(settings, processor="verify", overrides=overrides)
    walker = OrderFolderWalker(
        target_root=config.target_root,
        order_filename=config.order_filename,
    )
    store = BeatmapFolderStore(
        target_root=config.target_root,
        order_filename=config.order_filename,
    )
    return BeatmapVerifyExporter(
        walker=walker,
        store=store,
        verify_filename=config.verify_filename,
        failed_filename=config.failed_filename,
    )


def build_verify_exporter_from_config_or_default(
    config_path: Path | None = None,
) -> BeatmapVerifyExporter:
    return build_from_config_or_default(
        _build_verify_exporter_from_config,
        [_load_verify_exporter_config],
        config_path=config_path,
        default_builder=_build_verify_exporter_from_config,
    )


def build_beatmap_verify_exporter_from_config_or_default(
    config_path: Path | None = None,
) -> BeatmapVerifyExporter:
    return build_verify_exporter_from_config_or_default(config_path)


def main():
    exporter = build_verify_exporter_from_config_or_default()
    exporter.run(overwrite=False)


if __name__ == "__main__":
    main()
