"""Compatibility import for the retired order.txt walker."""

from Traning.Lib.beatmap.manifest import ManifestFolderWalker


OrderFolderWalker = ManifestFolderWalker

__all__ = ["ManifestFolderWalker", "OrderFolderWalker"]
