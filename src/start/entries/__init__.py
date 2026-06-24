from start.entries.before_traning import ENTRY as BEFORE_TRANING_ENTRY
from start.entries.package import ENTRY as PACKAGE_ENTRY
from start.entries.traning import ENTRY as TRANING_ENTRY
from start.modules import START_ENTRY

SRC_ENTRIES = (
    START_ENTRY,
    PACKAGE_ENTRY,
    BEFORE_TRANING_ENTRY,
    TRANING_ENTRY,
)

__all__ = [
    "BEFORE_TRANING_ENTRY",
    "PACKAGE_ENTRY",
    "SRC_ENTRIES",
    "START_ENTRY",
    "TRANING_ENTRY",
]
