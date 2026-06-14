from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Literal

from Traning.Lib.beatmap.manifest import ManifestFolderWalker
from Traning.Lib.common.pathspec import filter_files, gitwildmatch_spec
from Traning.state.manifest_schema import MANIFEST_DB_FILENAME


WriteMode = Literal["overwrite", "append", "skip_if_exists"]


class BeatmapFolderStore:
    """
    严格规则：
    1. 只允许操作 SQLite manifest 中启用的内部文件夹
    2. 不负责创建文件夹
    3. 文件夹不存在时直接报错，由 PackageUpdater 负责先同步目录结构
    """

    def __init__(
        self,
        target_root: str,
        manifest_filename: str = MANIFEST_DB_FILENAME,
    ):
        self.target_root = Path(target_root)
        self.manifest_filename = manifest_filename

        if not self.target_root.exists():
            raise FileNotFoundError(f"目录不存在: {self.target_root}")

        self.walker = ManifestFolderWalker(
            target_root=str(self.target_root),
            manifest_filename=self.manifest_filename,
        )

    def _normalize_folder_name(self, folder_name: str) -> str:
        folder_name = folder_name.strip()
        if not folder_name:
            raise ValueError("folder_name 不能为空")

        if Path(folder_name).name != folder_name:
            raise ValueError(f"folder_name 非法，不能包含路径层级: {folder_name}")

        return folder_name

    def _registered_names(self) -> set[str]:
        return set(self.walker.read_folder_names())

    def is_registered(self, folder_name: str) -> bool:
        folder_name = self._normalize_folder_name(folder_name)
        return folder_name in self._registered_names()

    def _assert_registered(self, folder_name: str):
        if not self.is_registered(folder_name):
            raise PermissionError(
                f"{folder_name} 未登记在 {self.target_root / self.manifest_filename} 中，不允许使用"
            )

    def get_folder_path(self, folder_name: str) -> Path:
        folder_name = self._normalize_folder_name(folder_name)
        self._assert_registered(folder_name)
        return self.target_root / folder_name

    def folder_exists(self, folder_name: str) -> bool:
        folder = self.get_folder_path(folder_name)
        return folder.exists()

    def _require_existing_folder(self, folder_name: str) -> Path:
        folder = self.get_folder_path(folder_name)
        if not folder.exists():
            raise FileNotFoundError(
                f"文件夹不存在: {folder}。请先调用 PackageUpdater.sync_folders_from_manifest()"
            )
        return folder

    def find_files(self, folder_name: str, pattern: str = "*") -> List[Path]:
        folder = self._require_existing_folder(folder_name)
        spec = gitwildmatch_spec((pattern.lower(),))
        return sorted(filter_files(folder.iterdir(), spec), key=lambda p: p.name.lower())

    def find_osu_files(self, folder_name: str) -> List[Path]:
        return self.find_files(folder_name, "*.osu")

    def get_file_path(self, folder_name: str, filename: str) -> Path:
        if not filename or Path(filename).name != filename:
            raise ValueError(f"filename 非法: {filename}")

        folder = self._require_existing_folder(folder_name)
        return folder / filename

    def file_exists(self, folder_name: str, filename: str) -> bool:
        return self.get_file_path(folder_name, filename).exists()

    def write_text(
        self,
        folder_name: str,
        filename: str,
        content: str,
        mode: WriteMode = "overwrite",
    ) -> str:
        """
        通用文本写入接口。

        参数:
            folder_name: 必须已登记在 manifest 中，且对应文件夹已存在
            filename: 目标文件名，例如 verify.txt
            content: 要写入的文本内容
            mode:
                - overwrite: 覆盖写入
                - append: 追加写入
                - skip_if_exists: 若文件已存在则跳过

        返回:
            "written" / "appended" / "skipped"
        """
        file_path = self.get_file_path(folder_name, filename)

        if mode == "skip_if_exists" and file_path.exists():
            return "skipped"

        if mode == "overwrite" or mode == "skip_if_exists":
            file_path.write_text(content, encoding="utf-8")
            return "written"

        if mode == "append":
            with file_path.open("a", encoding="utf-8") as f:
                f.write(content)
            return "appended"

        raise ValueError(f"不支持的写入模式: {mode}")

    def write_lines(
        self,
        folder_name: str,
        filename: str,
        lines: Iterable[str],
        mode: WriteMode = "overwrite",
        add_trailing_newline: bool = True,
    ) -> str:
        text = "\n".join(lines)
        if add_trailing_newline and text:
            text += "\n"

        return self.write_text(
            folder_name=folder_name,
            filename=filename,
            content=text,
            mode=mode,
        )

    def append_line(
        self,
        folder_name: str,
        filename: str,
        line: str,
    ) -> str:
        return self.write_text(
            folder_name=folder_name,
            filename=filename,
            content=f"{line}\n",
            mode="append",
        )

    def read_text(self, folder_name: str, filename: str) -> str:
        file_path = self.get_file_path(folder_name, filename)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return file_path.read_text(encoding="utf-8")
