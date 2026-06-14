from __future__ import annotations

import zipfile
from pathlib import Path
from time import perf_counter

from loguru import logger

from before_traning.Lib.beatmap.manifest import ManifestEntry
from before_traning.Lib.beatmap.osz import OsuEntry, read_osz_entry
from before_traning.Lib.beatmap.package import PackageUpdater
from before_traning.Lib.common.failures import format_exception
from before_traning.Lib.common.pathspec import suffix_spec
from before_traning.Lib.common.processing import matching_files
from before_traning.conf import DEFAULT_SETTINGS, Settings, load_settings
from before_traning.conf.field_groups import assign_group
from before_traning.conf.legacy_config import settings_namespace
from before_traning.state.process_status import ProcessStatusManager


class BeatmapImportProcessor:
    def __init__(
        self,
        settings: Settings = DEFAULT_SETTINGS,
        **overrides: object,
    ):
        if not isinstance(settings, Settings):
            overrides = {"export_dir": settings, **overrides}
            settings = DEFAULT_SETTINGS

        config = settings_namespace(
            settings,
            processor="beatmap_import",
            overrides=overrides,
        )
        assign_group(self, config, "beatmap_import")
        self.export_dir = Path(self.export_dir)
        self.target_root = Path(self.target_root)
        self.keyword = config.keyword.lower()
        self.osz_file_spec = suffix_spec((".osz",))
        self.updater = PackageUpdater(
            target_root=self.target_root,
            manifest_filename=self.manifest_filename,
            ignore_patterns=settings.package.ignore_patterns,
        )
        self.status_manager = ProcessStatusManager(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
        )
        self.success_count = 0
        self.skip_count = 0
        self.fail_count = 0

    def _scan_single_osz(self, osz_path: Path) -> OsuEntry | None:
        return read_osz_entry(
            osz_path,
            keyword=self.keyword,
            audio_output_filename=self.audio_filename,
        )

    def _scan_entries(self) -> list[OsuEntry]:
        osz_files = matching_files(
            self.export_dir,
            self.osz_file_spec,
            sort_key=lambda path: (
                path.stat().st_mtime_ns,
                path.name.lower(),
            ),
        )
        if not osz_files:
            print(f"没有在 {self.export_dir} 中找到 .osz 文件")
            return []

        entries: list[OsuEntry] = []
        seen_names: set[str] = set()
        for osz_path in osz_files:
            try:
                entry = self._scan_single_osz(osz_path)
            except zipfile.BadZipFile:
                print(f"[跳过] {osz_path.name}：不是有效压缩包")
                self.skip_count += 1
                continue
            except Exception as error:
                print(
                    f"[失败] {osz_path.name}：扫描失败："
                    f"{format_exception(error)}"
                )
                self.fail_count += 1
                continue

            if entry is None:
                print(
                    f"[跳过] {osz_path.name}："
                    f"未找到包含 '{self.keyword}' 的 .osu"
                )
                self.skip_count += 1
                continue
            if entry.osu_base_name in seen_names:
                raise ValueError(
                    f"扫描结果中出现重复目录名: {entry.osu_base_name} "
                    f"(来源文件至少包括 {osz_path.name})"
                )
            seen_names.add(entry.osu_base_name)
            entries.append(entry)
            print(
                f"[登记候选] {osz_path.name} -> {entry.osu_base_name} "
                f"(audio: {entry.audio_source_filename})"
            )

        entries.sort(key=lambda entry: entry.sort_key)
        return entries

    def _rebuild_manifest(self, entries: list[OsuEntry]) -> None:
        folder_names = self.updater.replace_manifest(
            [
                ManifestEntry(
                    source_name=entry.osu_base_name,
                    osu_filename=entry.osu_filename,
                    source_osz_name=entry.osz_path.name,
                    source_mtime_ns=entry.sort_key[0],
                )
                for entry in entries
            ]
        )
        for entry in entries:
            entry.folder_name = folder_names[entry.osu_base_name]
        print(f"[完成] 已更新 SQLite manifest，共 {len(entries)} 项")

    def _write_entries(self, entries: list[OsuEntry]) -> None:
        self.updater.sync_folders_from_manifest()
        for entry in entries:
            folder_name = entry.folder_name
            if folder_name is None:
                raise RuntimeError(f"{entry.osu_base_name} 尚未分配内部目录 ID")

            destination = self.updater.create_folder_if_registered(folder_name)
            (destination / entry.osu_filename).write_bytes(entry.osu_bytes)
            (destination / self.audio_filename).write_bytes(entry.audio_bytes)
            self.status_manager.ensure_status_file(folder_name)
            status_details = {
                "osu_imported": {
                    "source_name": entry.osu_base_name,
                    "osu_filename": entry.osu_filename,
                },
                "audio_imported": {
                    "source_name": entry.osu_base_name,
                    "source_audio_filename": entry.audio_source_filename,
                    "saved_audio_filename": self.audio_filename,
                },
            }
            for step, detail in status_details.items():
                self.status_manager.mark_step_done(
                    folder_name,
                    step,
                    detail=detail,
                )
            print(
                f"[完成] {entry.osz_path.name} -> {folder_name} "
                f"({entry.osu_base_name})"
            )
            self.success_count += 1

    def run(self) -> bool:
        entries = self._scan_entries()
        if not entries:
            print("没有可登记的目标 .osu")
            return self.fail_count == 0

        self._rebuild_manifest(entries)
        self._write_entries(entries)
        extra_directories = self.updater.find_unregistered_existing_folders()
        if extra_directories:
            print("\n警告：发现未登记到 manifest 的现有文件夹，这些文件夹不会被使用：")
            for path in extra_directories:
                print(f"  - {path}")

        print(
            f"\n处理完成：成功 {self.success_count} 个，"
            f"跳过 {self.skip_count} 个，失败 {self.fail_count} 个"
        )
        print(f"manifest：{self.updater.manifest_path}")
        print(f"谱面编号对照表：{self.updater.manifest_table_path}")
        return self.fail_count == 0


OsuOszProcessor = BeatmapImportProcessor


def build_beatmap_import_processor_from_config_or_default(
    config_path: Path | None = None,
) -> BeatmapImportProcessor:
    return BeatmapImportProcessor(load_settings(config_path))


def build_osu_osz_processor_from_config_or_default(
    config_path: Path | None = None,
) -> BeatmapImportProcessor:
    return build_beatmap_import_processor_from_config_or_default(config_path)


def import_beatmaps(settings: Settings) -> bool:
    logger.info("开始 import_beatmaps")
    started_at = perf_counter()
    success = BeatmapImportProcessor(settings).run()
    logger.info("完成 import_beatmaps ({:.2f}s)", perf_counter() - started_at)
    return success


__all__ = [
    "BeatmapImportProcessor",
    "OsuEntry",
    "OsuOszProcessor",
    "build_beatmap_import_processor_from_config_or_default",
    "build_osu_osz_processor_from_config_or_default",
    "import_beatmaps",
]
