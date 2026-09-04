"""可校验训练与数据制品的清单契约。"""

from dataclasses import dataclass

from .common import JSONValue, require_identifier, require_nonnegative, require_sha256
from .data import DataSplit


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """制品身份、来源和完整性信息。"""

    artifact_id: str
    artifact_type: str
    schema_version: int
    dataset_id: str
    split: DataSplit
    producer_id: str
    row_count: int
    sha256: str
    created_at_ms: float
    metadata: tuple[tuple[str, JSONValue], ...] = ()

    def __post_init__(self) -> None:
        for field_name, value in (
            ("artifact_id", self.artifact_id),
            ("artifact_type", self.artifact_type),
            ("dataset_id", self.dataset_id),
            ("producer_id", self.producer_id),
        ):
            require_identifier(value, field_name)
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema_version 必须是整数")
        if self.schema_version < 1:
            raise ValueError("schema_version 必须至少为 1")
        if not isinstance(self.split, DataSplit):
            raise TypeError("split 必须是 canonical DataSplit")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int):
            raise TypeError("row_count 必须是整数")
        if self.row_count < 0:
            raise ValueError("row_count 不得为负数")
        require_sha256(self.sha256)
        require_nonnegative(self.created_at_ms, "created_at_ms")
        if not isinstance(self.metadata, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in self.metadata
        ):
            raise TypeError("metadata 必须是二元 tuple 的 tuple")
        keys = tuple(key for key, _ in self.metadata)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("metadata 的键必须是字符串")
        if any(not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("metadata 的键必须非空且唯一")
