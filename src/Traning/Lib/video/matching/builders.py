from __future__ import annotations

from pathlib import Path

from Traning.Lib.common.batch import read_config_values
from Traning.conf.legacy_config import (
    ConfigReader,
    VIDEO_PACKAGE_RENAMER_CONFIG_SPECS,
    build_from_config_or_default,
)
from Traning.Lib.video.matching.renamer import VideoPackageRenamer


def _load_video_package_renamer_config(config: ConfigReader) -> dict[str, object]:
    return read_config_values(config, VIDEO_PACKAGE_RENAMER_CONFIG_SPECS)


def build_video_package_renamer_from_config_or_default(
    config_path: Path | None = None,
) -> VideoPackageRenamer:
    return build_from_config_or_default(
        VideoPackageRenamer,
        [_load_video_package_renamer_config],
        config_path=config_path,
        default_builder=VideoPackageRenamer,
    )
