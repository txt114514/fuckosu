"""公开 typed repository 契约及其内存、SQLite adapters。"""

from .memory import (
    InMemoryDatasetCatalogRepository,
    InMemoryPreprocessingMetadataRepository,
)
from .models import DatasetCatalogEntry, PreprocessingMetadata, RepositoryError
from .protocols import DatasetCatalogRepository, PreprocessingMetadataRepository
from .sqlite import (
    SQLiteDatasetCatalogRepository,
    SQLitePreprocessingMetadataRepository,
)

__all__ = (
    "DatasetCatalogEntry",
    "DatasetCatalogRepository",
    "InMemoryDatasetCatalogRepository",
    "InMemoryPreprocessingMetadataRepository",
    "PreprocessingMetadata",
    "PreprocessingMetadataRepository",
    "RepositoryError",
    "SQLiteDatasetCatalogRepository",
    "SQLitePreprocessingMetadataRepository",
)
