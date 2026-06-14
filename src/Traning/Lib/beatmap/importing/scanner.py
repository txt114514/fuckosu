from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from Traning.Lib.beatmap.importing.entry import OsuEntry
from Traning.Lib.beatmap.osu_metadata import read_audio_filename
from Traning.Lib.common.failures import format_exception
from Traning.Lib.common.pathspec import filter_files, matches_name


class OszScannerMixin:
    def _is_target_osu(self, path: Path) -> bool:
        return (
            path.is_file()
            and matches_name(self.osu_file_spec, path)
            and self.keyword in path.name.lower()
        )

    def _read_audio_bytes(self, osu_path: Path) -> tuple[str, bytes]:
        audio_filename = read_audio_filename(osu_path)
        audio_path = (osu_path.parent / audio_filename).resolve()

        if not audio_path.is_file():
            raise FileNotFoundError(f"{osu_path} 对应音频不存在: {audio_filename}")

        if not matches_name(self.mp3_file_spec, audio_path):
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
        if not self.export_dir.exists():
            print(f"没有在 {self.export_dir} 中找到 .osz 文件")
            return []

        osz_files = sorted(
            filter_files(self.export_dir.iterdir(), self.osz_file_spec),
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
                print(f"[失败] {osz_path.name}：扫描失败：{format_exception(e)}")
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
