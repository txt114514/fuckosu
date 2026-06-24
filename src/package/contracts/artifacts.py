from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from package.contracts.base import ContractMixin


@dataclass(frozen=True)
class ArtifactFileRef(ContractMixin):
    role: str
    path: str
    sha256: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("artifact role must not be empty")
        if not self.path:
            raise ValueError("artifact path must not be empty")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be nonnegative")

    @classmethod
    def from_path(
        cls,
        role: str,
        path: Path,
        *,
        sha256: str | None = None,
    ) -> ArtifactFileRef:
        size = path.stat().st_size if path.exists() and path.is_file() else None
        return cls(
            role=role,
            path=path.as_posix(),
            sha256=sha256,
            size_bytes=size,
        )


@dataclass(frozen=True)
class VersionedArtifactRef(ContractMixin):
    artifact_id: str
    schema_version: str
    files: tuple[ArtifactFileRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must not be empty")
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        object.__setattr__(
            self,
            "files",
            tuple(
                item
                if isinstance(item, ArtifactFileRef)
                else ArtifactFileRef.from_mapping(item)
                for item in self.files
            ),
        )


__all__ = ["ArtifactFileRef", "VersionedArtifactRef"]
