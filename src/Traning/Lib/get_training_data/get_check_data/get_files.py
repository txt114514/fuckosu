from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import zipfile

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from Traning.Lib.function_tools.functions_process_tool import (
    read_config_values,
)
from Traning.Lib.get_training_data.config_loader import (
    build_from_config_or_default,
    ConfigReader,
    OSU_OSZ_PROCESSOR_CONFIG_SPECS,
)
from Traning.Lib.traning_package_manager.package_update import PackageUpdater
from Traning.Lib.get_training_data.process_status_manager import (
    ProcessStatusManager,
)

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_EXPORT_DIR = DEFAULT_REPO_ROOT / "osu-lazer" / "exports"
DEFAULT_TARGET_ROOT = DEFAULT_REPO_ROOT / "training_package" / "match-completed_package"
DEFAULT_KEYWORD = "normal"
DEFAULT_ORDER_FILENAME = "order.txt"
DEFAULT_AUDIO_FILENAME = "audio.mp3"

# 默认值保留在当前文件；config.json 里的合法参数只用于覆盖这些默认值。


def _load_osu_osz_processor_config(config: ConfigReader) -> dict[str, object]:
    # get_files 只关心导入来源、目标目录、选谱关键字和导出音频文件名。
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


@dataclass
class OsuEntry:
    osz_path: Path
    osu_base_name: str
    osu_filename: str
    osu_bytes: bytes
    audio_source_filename: str
    audio_bytes: bytes
    sort_key: tuple[int, str]


class OsuOszProcessor:
    """
    严格规则：
    1. 所有登记顺序以 .osz 的时间排序为准
    2. order.txt 每次按扫描结果重建
    3. 只有在 order.txt 中出现的文件夹才会被使用
    """

    def __init__(
        self,
        export_dir: str = str(DEFAULT_EXPORT_DIR),
        target_root: str = str(DEFAULT_TARGET_ROOT),
        keyword: str = DEFAULT_KEYWORD,
        order_filename: str = DEFAULT_ORDER_FILENAME,
        audio_filename: str = DEFAULT_AUDIO_FILENAME,
    ):
        self.export_dir = Path(export_dir)
        self.target_root = Path(target_root)
        self.keyword = keyword.lower()
        self.order_filename = order_filename
        self.audio_filename = audio_filename
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

    def _is_target_osu(self, path: Path) -> bool:
        return (
            path.is_file()
            and path.suffix.lower() == ".osu"
            and self.keyword in path.name.lower()
        )

    def _extract_audio_filename(self, osu_path: Path) -> str:
        in_general_section = False

        with osu_path.open("r", encoding="utf-8-sig") as f:
            for raw_line in f:
                line = raw_line.strip()

                if not line or line.startswith("//"):
                    continue

                if line.startswith("[") and line.endswith("]"):
                    in_general_section = (line == "[General]")
                    continue

                if not in_general_section or ":" not in line:
                    continue

                key, value = line.split(":", 1)
                if key.strip() == "AudioFilename":
                    audio_filename = value.strip()
                    if not audio_filename:
                        raise ValueError(f"{osu_path} 的 AudioFilename 为空")
                    return audio_filename

        raise ValueError(f"{osu_path} 缺少 AudioFilename")

    def _read_audio_bytes(self, osu_path: Path) -> tuple[str, bytes]:
        audio_filename = self._extract_audio_filename(osu_path)
        audio_path = (osu_path.parent / audio_filename).resolve()

        if not audio_path.is_file():
            raise FileNotFoundError(f"{osu_path} 对应音频不存在: {audio_filename}")

        if audio_path.suffix.lower() != ".mp3":
            raise ValueError(
                f"{osu_path} 对应音频不是 mp3，当前仅支持导出为 {self.audio_filename}: "
                f"{audio_filename}"
            )

        return audio_path.name, audio_path.read_bytes()

    def _scan_single_osz(self, osz_path: Path) -> OsuEntry | None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            with zipfile.ZipFile(osz_path, "r") as zf:
                zf.extractall(tmp_path)

            matched_osu_files = sorted(
                [p for p in tmp_path.rglob("*") if self._is_target_osu(p)],
                key=lambda p: p.name.lower(),
            )

            if not matched_osu_files:
                return None

            chosen_osu = matched_osu_files[0]
            audio_source_filename, audio_bytes = self._read_audio_bytes(chosen_osu)

            return OsuEntry(
                osz_path=osz_path,
                osu_base_name=chosen_osu.stem,
                osu_filename=chosen_osu.name,
                osu_bytes=chosen_osu.read_bytes(),
                audio_source_filename=audio_source_filename,
                audio_bytes=audio_bytes,
                sort_key=(osz_path.stat().st_mtime_ns, osz_path.name.lower()),
            )

    def _scan_all_entries_in_time_order(self) -> list[OsuEntry]:
        osz_files = sorted(
            self.export_dir.glob("*.osz"),
            key=lambda p: (p.stat().st_mtime_ns, p.name.lower()),
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
            except Exception as e:
                print(f"[失败] {osz_path.name}：扫描失败：{e}")
                self.fail_count += 1
                continue

            if entry is None:
                print(f"[跳过] {osz_path.name}：未找到包含 '{self.keyword}' 的 .osu")
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

        entries.sort(key=lambda e: e.sort_key)
        return entries

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

    def run(self):
        entries = self._scan_all_entries_in_time_order()
        if not entries:
            print("没有可登记的目标 .osu")
            return

        # 严格时间顺序：先重建 order.txt
        self._rebuild_order(entries)

        # 再只使用 order.txt 中允许的文件夹
        self._sync_folders_and_copy_files(entries)

        extra_dirs = self.updater.find_unregistered_existing_folders()
        if extra_dirs:
            print()
            print("警告：发现未登记到 order.txt 的现有文件夹，这些文件夹不会被使用：")
            for p in extra_dirs:
                print(f"  - {p}")

        print()
        print(
            f"处理完成：成功 {self.success_count} 个，跳过 {self.skip_count} 个，失败 {self.fail_count} 个"
        )
        print(f"记录文件：{self.updater.order_file}")

def main():
    processor = build_osu_osz_processor_from_config_or_default()
    processor.run()


if __name__ == "__main__":
    main()
