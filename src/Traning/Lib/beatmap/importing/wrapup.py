from __future__ import annotations


class OszImportWrapUpMixin:
    def run(self) -> bool:
        entries = self._scan_all_entries_in_time_order()
        if not entries:
            print("没有可登记的目标 .osu")
            return self.fail_count == 0

        self._rebuild_manifest(entries)

        self._sync_folders_and_copy_files(entries)

        extra_dirs = self.updater.find_unregistered_existing_folders()
        if extra_dirs:
            print()
            print("警告：发现未登记到 manifest 的现有文件夹，这些文件夹不会被使用：")
            for p in extra_dirs:
                print(f"  - {p}")

        print()
        print(
            f"处理完成：成功 {self.success_count} 个，跳过 {self.skip_count} 个，失败 {self.fail_count} 个"
        )
        print(f"manifest：{self.updater.manifest_path}")
        print(f"谱面编号对照表：{self.updater.manifest_table_path}")
        return self.fail_count == 0
