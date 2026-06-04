from __future__ import annotations

from pathlib import Path
from typing import List


class OrderFolderWalker:
    def __init__(self, target_root: str, order_filename: str = "order.txt"):
        self.target_root = Path(target_root)
        self.order_file = self.target_root / order_filename

        if not self.target_root.exists():
            raise FileNotFoundError(f"目录不存在: {self.target_root}")
        if not self.order_file.exists():
            raise FileNotFoundError(f"索引文件不存在: {self.order_file}")

    def read_folder_names(self) -> List[str]:
        lines = self.order_file.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]