from __future__ import annotations

from pathlib import Path


def read_section_key(osu_path: Path, section_name: str, key_name: str) -> str:
    expected_section = f"[{section_name}]"
    in_section = False

    with osu_path.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                in_section = (line == expected_section)
                continue

            if not in_section or ":" not in line:
                continue

            key, value = line.split(":", 1)
            if key.strip() == key_name:
                stripped_value = value.strip()
                if not stripped_value:
                    raise ValueError(f"{osu_path} 的 {key_name} 为空")
                return stripped_value

    raise ValueError(f"{osu_path} 缺少 {key_name}")


def read_audio_filename(osu_path: Path) -> str:
    return read_section_key(osu_path, "General", "AudioFilename")


def read_overall_difficulty(osu_path: Path) -> float:
    return float(read_section_key(osu_path, "Difficulty", "OverallDifficulty"))
