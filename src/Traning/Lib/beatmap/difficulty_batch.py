from __future__ import annotations


class DifficultyBatchMixin:
    def run(self, overwrite: bool = False):
        folder_names = self.walker.read_folder_names()

        for folder_name in folder_names:
            try:
                result = self.export_one(folder_name, overwrite=overwrite)
                if result == "success":
                    print(f"[完成] {folder_name}")
                else:
                    print(f"[跳过] {folder_name}")
            except Exception as e:
                self.fail_count += 1
                self.failed_cases.append((folder_name, str(e)))
                if self.store.folder_exists(folder_name):
                    self.status_manager.ensure_status_file(folder_name)
                    self.status_manager.mark_step_pending(
                        folder_name,
                        "difficulty_exported",
                        detail={"error": str(e)},
                    )
                print(f"[失败] {folder_name}: {e}")

        failed_path = self.store.write_failed_report(
            self.failed_cases,
            failed_filename=self.failed_filename,
        )

        print()
        print(
            f"处理完成：成功 {self.success_count} 个，跳过 {self.skip_count} 个，失败 {self.fail_count} 个"
        )
        print(f"失败名单：{failed_path}")
