from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
