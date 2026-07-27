"""保留已停用 order.txt 目录遍历器名称的兼容导入。"""

from before_traning.Lib.beatmap.manifest import ManifestFolderWalker


OrderFolderWalker = ManifestFolderWalker

__all__ = ["ManifestFolderWalker", "OrderFolderWalker"]
