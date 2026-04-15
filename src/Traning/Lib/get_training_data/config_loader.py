from __future__ import annotations

import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


class CheckDataConfigError(Exception):
    pass


def _resolve_config_path(base_dir: Path, value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def load_check_data_config(config_path: Path | None = None) -> dict[str, str]:
    config_path = config_path or CONFIG_PATH
    base_dir = config_path.parent

    if not config_path.exists():
        raise CheckDataConfigError(f"配置文件不存在: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = json.load(f)
    except json.JSONDecodeError as e:
        raise CheckDataConfigError(f"{config_path} JSON 格式错误: {e}") from e

    section = raw_config.get("check_data")
    if not isinstance(section, dict):
        raise CheckDataConfigError(f"{config_path} 缺少 check_data 配置段")

    required_fields = [
        "export_dir",
        "target_root",
        "keyword",
    ]

    missing_fields = [field for field in required_fields if field not in section]
    if missing_fields:
        raise CheckDataConfigError(
            f"{config_path} 缺少配置字段: {', '.join(missing_fields)}"
        )

    return {
        "export_dir": _resolve_config_path(base_dir, str(section["export_dir"])),
        "target_root": _resolve_config_path(base_dir, str(section["target_root"])),
        "keyword": str(section["keyword"]),
    }
