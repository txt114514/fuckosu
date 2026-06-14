from __future__ import annotations

from sqlmodel import Field, SQLModel, UniqueConstraint


SEGMENT_DB_FILENAME = ".segment_manifest.sqlite"


class SegmentDatasetItem(SQLModel, table=True):
    __tablename__ = "segment_dataset_item"
    __table_args__ = (
        UniqueConstraint(
            "folder_name",
            "segment_id",
            name="uq_segment_folder_id",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    folder_name: str = Field(index=True)
    segment_id: str = Field(index=True)
    sequence: int = Field(index=True)
    dataset_dimension: str = Field(index=True)
    category: str = Field(index=True)
    video_path: str
    beatmap_path: str
    row_json: str
