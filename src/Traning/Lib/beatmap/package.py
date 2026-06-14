from __future__ import annotations

from pathlib import Path
from typing import List

import pathspec

from Traning.Lib.beatmap.manifest import ManifestEntry, PackageManifest
from Traning.conf import Settings
from Traning.conf.legacy_config import settings_namespace
from Traning.Lib.defaults import DEFAULT_SETTINGS as DEFAULTS


class PackageUpdater:
    """
    规则：
    1. SQLite manifest 是唯一可信索引
    2. 文件夹使用稳定的内部 ID
    3. 处理顺序只保存在 manifest 中
    """

    def __init__(
        self,
        target_root: str | Settings,
        **overrides: object,
    ):
        settings = target_root if isinstance(target_root, Settings) else DEFAULTS
        if not isinstance(target_root, Settings):
            overrides = {"target_root": target_root, **overrides}

        config = settings_namespace(settings, processor="package", overrides=overrides)
        self.target_root = Path(config.target_root)
        self.manifest_filename = config.manifest_filename
        self.ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", config.ignore_patterns)

        self.target_root.mkdir(parents=True, exist_ok=True)
        self.manifest = PackageManifest(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
        )
        self.manifest_path = self.manifest.db_path
        self.manifest_table_path = self.manifest.table_path

    def load_manifest_folder_names(self) -> List[str]:
        return self.manifest.read_folder_names()

    def load_registered_names(self) -> set[str]:
        return set(self.manifest.read_all_folder_names())

    def replace_manifest(self, entries: list[ManifestEntry]) -> dict[str, str]:
        return self.manifest.replace(entries)

    def is_registered(self, folder_name: str) -> bool:
        return self.manifest.is_active(folder_name)

    def create_folder_if_registered(self, folder_name: str) -> Path:
        """
        只有在 manifest 中启用的内部 ID 才允许创建/使用对应文件夹。
        """
        folder_name = folder_name.strip()
        if not folder_name:
            raise ValueError("folder_name 不能为空")

        if not self.is_registered(folder_name):
            raise PermissionError(
                f"{folder_name} 未登记到 {self.manifest_path}，不允许创建或使用该文件夹"
            )

        folder_path = self.target_root / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def sync_folders_from_manifest(self) -> List[Path]:
        """Create active manifest folders in processing order."""
        folder_names = self.load_manifest_folder_names()
        result: List[Path] = []

        for name in folder_names:
            folder_path = self.target_root / name
            folder_path.mkdir(parents=True, exist_ok=True)
            result.append(folder_path)

        return result

    def find_unregistered_existing_folders(self) -> List[Path]:
        """
        返回 target_root 下存在，但不在 manifest 中登记的目录。
        这些目录按你的规则“不应该被使用”。
        """
        registered = self.load_registered_names()
        extra_dirs: List[Path] = []

        for p in sorted(self.target_root.iterdir(), key=lambda x: x.name.lower()):
            if not p.is_dir():
                continue
            if self.ignore_spec.match_file(p.name + "/"):
                continue
            if p.name not in registered:
                extra_dirs.append(p)

        return extra_dirs
