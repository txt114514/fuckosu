from __future__ import annotations

from Traning.Lib.beatmap.importing.entry import OsuEntry


class OszImportWriterMixin:
    def _rebuild_order(self, entries: list[OsuEntry]):
        ordered_names = [e.osu_base_name for e in entries]
        self.updater.overwrite_order(ordered_names)
        print(f"[完成] 已重建 order.txt，共 {len(ordered_names)} 项")

    def _sync_folders_and_copy_files(self, entries: list[OsuEntry]):
        # 只按 order.txt 创建允许使用的文件夹
        self.updater.sync_folders_from_order()

        registered = self.updater.load_registered_names()

        for entry in entries:
            if entry.osu_base_name not in registered:
                raise PermissionError(
                    f"{entry.osu_base_name} 未登记在 order.txt 中，拒绝使用该文件夹"
                )

            dest_dir = self.updater.create_folder_if_registered(entry.osu_base_name)
            dest_osu_file = dest_dir / entry.osu_filename
            dest_audio_file = dest_dir / self.audio_filename

            # 这里用覆盖写入，保证当前包内容严格对应当前重建后的顺序结果
            dest_osu_file.write_bytes(entry.osu_bytes)
            dest_audio_file.write_bytes(entry.audio_bytes)

            self.status_manager.ensure_status_file(entry.osu_base_name)
            self.status_manager.mark_step_done(
                entry.osu_base_name,
                "osu_imported",
                detail={"osu_filename": entry.osu_filename},
            )
            self.status_manager.mark_step_done(
                entry.osu_base_name,
                "audio_imported",
                detail={
                    "source_audio_filename": entry.audio_source_filename,
                    "saved_audio_filename": self.audio_filename,
                },
            )

            print(
                f"[完成] {entry.osz_path.name} -> {dest_osu_file} + {dest_audio_file.name}"
            )
            self.success_count += 1
