"""Compatibility import for the retired order.txt walker."""

from before_traning.Lib.beatmap.manifest import ManifestFolderWalker


OrderFolderWalker = ManifestFolderWalker

__all__ = ["ManifestFolderWalker", "OrderFolderWalker"]
