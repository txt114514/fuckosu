from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from before_traning.Lib.beatmap.osu_metadata import read_audio_filename


@dataclass
class OsuEntry:
    osz_path: Path
    osu_base_name: str
    osu_filename: str
    osu_bytes: bytes
    audio_source_filename: str
    audio_bytes: bytes
    sort_key: tuple[int, str]
    folder_name: str | None = None


def read_osz_entry(
    osz_path: Path,
    *,
    keyword: str,
    audio_output_filename: str,
) -> OsuEntry | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(osz_path, "r") as archive:
            archive.extractall(tmp_path)
        matched_osu_files = sorted(
            (
                path
                for path in tmp_path.rglob("*")
                if path.is_file()
                and path.suffix.lower() == ".osu"
                and keyword.lower() in path.name.lower()
            ),
            key=lambda path: path.name.lower(),
        )
        if not matched_osu_files:
            return None

        chosen_osu = matched_osu_files[0]
        audio_filename = read_audio_filename(chosen_osu)
        audio_path = (chosen_osu.parent / audio_filename).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"{chosen_osu} 对应音频不存在: {audio_filename}"
            )
        if audio_path.suffix.lower() != ".mp3":
            raise ValueError(
                f"{chosen_osu} 对应音频不是 mp3，当前仅支持导出为 "
                f"{audio_output_filename}: {audio_filename}"
            )
        return OsuEntry(
            osz_path=osz_path,
            osu_base_name=chosen_osu.stem,
            osu_filename=chosen_osu.name,
            osu_bytes=chosen_osu.read_bytes(),
            audio_source_filename=audio_path.name,
            audio_bytes=audio_path.read_bytes(),
            sort_key=(osz_path.stat().st_mtime_ns, osz_path.name.lower()),
        )


__all__ = ["OsuEntry", "read_osz_entry"]
