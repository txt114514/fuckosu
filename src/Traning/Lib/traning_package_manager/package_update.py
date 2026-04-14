from __future__ import annotations

from pathlib import Path
from typing import List

from Traning.Lib.traning_package_manager.order_walker import OrderFolderWalker


class PackageUpdater:
    """
    规则：
    1. order.txt 是唯一可信索引
    2. 文件夹只有在 order.txt 中登记后才允许被使用
    3. 文件夹创建顺序以 order.txt 为准
    """

    def __init__(self, target_root: str, order_filename: str = "order.txt"):
        self.target_root = Path(target_root)
        self.order_filename = order_filename
        self.order_file = self.target_root / order_filename

        self.target_root.mkdir(parents=True, exist_ok=True)
        if not self.order_file.exists():
            self.order_file.touch()

    def load_order_list(self) -> List[str]:
        walker = OrderFolderWalker(
            target_root=str(self.target_root),
            order_filename=self.order_filename,
        )
        return walker.read_folder_names()

    def load_registered_names(self) -> set[str]:
        return set(self.load_order_list())

    def overwrite_order(self, folder_names: List[str]):
        """
        直接按给定顺序重写 order.txt。
        这是严格时间排序模式下最关键的方法。
        """
        cleaned: List[str] = []
        seen = set()

        for name in folder_names:
            name = name.strip()
            if not name:
                continue
            if name in seen:
                raise ValueError(f"order 中出现重复目录名: {name}")
            seen.add(name)
            cleaned.append(name)

        text = "\n".join(cleaned)
        if text:
            text += "\n"

        self.order_file.write_text(text, encoding="utf-8")

    def is_registered(self, folder_name: str) -> bool:
        return folder_name in self.load_registered_names()

    def create_folder_if_registered(self, folder_name: str) -> Path:
        """
        只有在 order.txt 中已登记的名字才允许创建/使用对应文件夹。
        """
        folder_name = folder_name.strip()
        if not folder_name:
            raise ValueError("folder_name 不能为空")

        if not self.is_registered(folder_name):
            raise PermissionError(
                f"{folder_name} 未登记到 {self.order_file}，不允许创建或使用该文件夹"
            )

        folder_path = self.target_root / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def sync_folders_from_order(self) -> List[Path]:
        """
        严格按 order.txt 同步文件夹。
        不在 order.txt 中的文件夹不会被这里使用。
        """
        folder_names = self.load_order_list()
        result: List[Path] = []

        for name in folder_names:
            folder_path = self.target_root / name
            folder_path.mkdir(parents=True, exist_ok=True)
            result.append(folder_path)

        return result

    def find_unregistered_existing_folders(self) -> List[Path]:
        """
        返回 target_root 下存在，但不在 order.txt 中登记的目录。
        这些目录按你的规则“不应该被使用”。
        """
        registered = self.load_registered_names()
        extra_dirs: List[Path] = []

        for p in sorted(self.target_root.iterdir(), key=lambda x: x.name.lower()):
            if not p.is_dir():
                continue
            if p.name not in registered:
                extra_dirs.append(p)

        return extra_dirs