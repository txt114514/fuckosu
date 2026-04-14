from pathlib import Path
import zipfile
import tempfile
import shutil


class OsuOszProcessor:
    def __init__(self, export_dir: str = "/home/dev/workspace/osu-lazer/exports",
                target_root: str = "/home/dev/workspace/training_package/match-completed_package",
                keyword: str = "normal"):
        self.export_dir = Path(export_dir)
        self.target_root = Path(target_root)
        self.order_log = self.target_root / "order.txt"
        self.keyword = keyword.lower()

        self.target_root.mkdir(parents=True, exist_ok=True)

        # 已登记的 .osu 名
        self.registered_set = self._load_registered_names()

        self.success_count = 0
        self.skip_count = 0

    def _is_target_osu(self, path: Path) -> bool:
        return (
            path.is_file()
            and path.suffix.lower() == ".osu"
            and self.keyword in path.name.lower()
        )

    def _load_registered_names(self) -> set[str]:
        if not self.order_log.exists():
            return set()

        lines = self.order_log.read_text(encoding="utf-8").splitlines()
        return {line.strip() for line in lines if line.strip()}

    def _append_order(self, name: str):
        with self.order_log.open("a", encoding="utf-8") as f:
            if self.order_log.exists() and self.order_log.stat().st_size > 0:
                f.write("\n")
            f.write(name)

    def _process_single_osz(self, osz_path: Path):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            with zipfile.ZipFile(osz_path, "r") as zf:
                zf.extractall(tmp_path)

            matched_osu_files = sorted(
                [p for p in tmp_path.rglob("*") if self._is_target_osu(p)],
                key=lambda p: p.name.lower()
            )

            if not matched_osu_files:
                print(f"[跳过] {osz_path.name}：未找到包含 '{self.keyword}' 的 .osu")
                self.skip_count += 1
                return

            chosen_osu = matched_osu_files[0]
            osu_base_name = chosen_osu.stem

            # 已处理过 → 跳过
            if osu_base_name in self.registered_set:
                print(f"[跳过] {osz_path.name}：{osu_base_name} 已登记")
                self.skip_count += 1
                return

            dest_dir = self.target_root / osu_base_name

            # 冲突检测
            if dest_dir.exists():
                raise FileExistsError(
                    f"冲突：目录已存在但未登记 → {dest_dir}"
                )

            dest_dir.mkdir(parents=True, exist_ok=False)

            dest_file = dest_dir / chosen_osu.name
            shutil.copy2(chosen_osu, dest_file)

            # 记录
            self._append_order(osu_base_name)
            self.registered_set.add(osu_base_name)

            print(f"[完成] {osz_path.name} -> {dest_file}")
            self.success_count += 1

            if len(matched_osu_files) > 1:
                print(f"       注意：多个匹配，仅使用 {chosen_osu.name}")

    def run(self):
        osz_files = sorted(
            self.export_dir.glob("*.osz"),
            key=lambda p: p.stat().st_mtime_ns
        )

        if not osz_files:
            print(f"没有在 {self.export_dir} 中找到 .osz 文件")
            return

        for osz_path in osz_files:
            try:
                self._process_single_osz(osz_path)
            except zipfile.BadZipFile:
                print(f"[错误] {osz_path.name} 不是有效压缩包")
                self.skip_count += 1
            except Exception as e:
                print(f"[错误] {osz_path.name} 处理失败：{e}")
                raise

        print()
        print(f"处理完成：成功 {self.success_count} 个，跳过/失败 {self.skip_count} 个")
        print(f"记录文件：{self.order_log}")
def main():
    processor = OsuOszProcessor(
        export_dir="/home/dev/workspace/osu-lazer/exports",
        target_root="/home/dev/workspace/training_package/match-completed_package",
        keyword="normal"
    )
    processor.run()


if __name__ == "__main__":
    main()