from __future__ import annotations

from Traning.Lib.beatmap.importing.entry import OsuEntry
from Traning.Lib.beatmap.manifest import ManifestEntry


class OszImportWriterMixin:
    def _rebuild_manifest(self, entries: list[OsuEntry]):
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

    def _sync_folders_and_copy_files(self, entries: list[OsuEntry]):
        self.updater.sync_folders_from_manifest()

        registered = self.updater.load_registered_names()

        for entry in entries:
            if entry.folder_name is None:
                raise RuntimeError(f"{entry.osu_base_name} 尚未分配内部目录 ID")
            if entry.folder_name not in registered:
                raise PermissionError(
                    f"{entry.folder_name} 未登记在 manifest 中，拒绝使用该文件夹"
                )

            dest_dir = self.updater.create_folder_if_registered(entry.folder_name)
            dest_osu_file = dest_dir / entry.osu_filename
            dest_audio_file = dest_dir / self.audio_filename

            dest_osu_file.write_bytes(entry.osu_bytes)
            dest_audio_file.write_bytes(entry.audio_bytes)

            self.status_manager.ensure_status_file(entry.folder_name)
            self.status_manager.mark_step_done(
                entry.folder_name,
                "osu_imported",
                detail={
                    "source_name": entry.osu_base_name,
                    "osu_filename": entry.osu_filename,
                },
            )
            self.status_manager.mark_step_done(
                entry.folder_name,
                "audio_imported",
                detail={
                    "source_name": entry.osu_base_name,
                    "source_audio_filename": entry.audio_source_filename,
                    "saved_audio_filename": self.audio_filename,
                },
            )

            print(
                f"[完成] {entry.osz_path.name} -> {entry.folder_name} "
                f"({entry.osu_base_name})"
            )
            self.success_count += 1
