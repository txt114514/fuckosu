from __future__ import annotations

from sqlmodel import Field, SQLModel, UniqueConstraint


MANIFEST_DB_FILENAME = ".package_manifest.sqlite"


class PackageManifestItem(SQLModel, table=True):
    __tablename__ = "package_manifest_item"
    __table_args__ = (
        UniqueConstraint("folder_name", name="uq_manifest_folder_name"),
        UniqueConstraint("source_name", name="uq_manifest_source_name"),
    )

    id: int | None = Field(default=None, primary_key=True)
    folder_name: str = Field(index=True)
    source_name: str = Field(index=True)
    sequence: int = Field(index=True)
    osu_filename: str | None = None
    source_osz_name: str | None = None
    source_mtime_ns: int | None = None
    difficulty_value: float | None = None
    active: bool = Field(default=True, index=True)
