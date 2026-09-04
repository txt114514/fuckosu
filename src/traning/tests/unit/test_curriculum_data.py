"""验证生产 curriculum 对真实 segment 元数据执行累计数据选择。"""

from __future__ import annotations

from pathlib import Path

import pytest
from package import CurriculumStage

from traning.state import DataSplit
from traning.core.data import SegmentTrainingDataset
from traning.lib.data.annotation import SegmentAnnotation
from traning.lib.data.models import SegmentRecord
from traning.core.training import (
    CurriculumDataUnavailableError,
    require_curriculum_dataset,
    select_curriculum_dataset,
)
from traning.core.training.production_stages import _applicable_frame_weights


_EXPECTED_KEYS = {
    CurriculumStage.BASIC: ("single", "slider"),
    CurriculumStage.MULTI_OBJECT: ("single", "slider", "multi"),
    CurriculumStage.COMPLEX: (
        "single",
        "slider",
        "multi",
        "point-slider",
        "spinner",
    ),
    CurriculumStage.FULL: (
        "single",
        "slider",
        "multi",
        "point-slider",
        "spinner",
        "long",
    ),
}


def _record(
    tmp_path: Path,
    *,
    key: str,
    dimension: str,
    category: str,
) -> SegmentRecord:
    """构造只需元数据即可展开帧引用的 segment。"""

    directory = tmp_path / key
    annotation = SegmentAnnotation(
        schema_version=1,
        segment_id=key,
        dataset_dimension=dimension,
        category=category,
        difficulty={
            "approach_preempt_ms": 600.0,
            "circle_radius_osu_pixels": 32.0,
        },
        source={
            "folder_name": "fixture",
            "osu_filename": "fixture.osu",
            "clip_start_ms": 0,
            "clip_end_ms": 100,
        },
        hit_objects=(),
    )
    return SegmentRecord(
        key=key,
        item_name="item-1",
        category=category,
        dataset_dimension=dimension,
        directory=directory,
        video_path=directory / "unused.mp4",
        annotation_path=directory / "beatmap.json",
        annotation=annotation,
    )


def _dataset(tmp_path: Path) -> SegmentTrainingDataset:
    """覆盖当前仓库六类生产 segment 的惰性数据集。"""

    specifications = (
        ("single", "atomic", "single_point"),
        ("slider", "atomic", "slider"),
        ("multi", "atomic", "multi_point"),
        ("point-slider", "atomic", "point_slider"),
        ("spinner", "atomic", "spinner"),
        ("long", "long_sequence", "long_sequence"),
    )
    return SegmentTrainingDataset(
        tuple(
            _record(
                tmp_path,
                key=key,
                dimension=dimension,
                category=category,
            )
            for key, dimension, category in specifications
        ),
        split=DataSplit.TRAIN,
        sample_fps=10.0,
        frame_step=1,
        max_frames_per_segment=None,
        visibility_post_ms=100.0,
        coordinate_transform=None,
    )


@pytest.mark.parametrize("stage", tuple(CurriculumStage))
def test_curriculum_data_is_cumulative_and_preserves_dataset_contract(
    tmp_path: Path,
    stage: CurriculumStage,
) -> None:
    """课程阶段只能增加 segment，且不得篡改 split 与采样规格。"""

    source = _dataset(tmp_path)
    selected = select_curriculum_dataset(source, stage)

    assert tuple(record.key for record in selected.records) == _EXPECTED_KEYS[stage]
    assert selected.split is source.split
    assert selected.sample_fps == source.sample_fps
    assert selected.frame_step == source.frame_step
    assert selected.max_frames_per_segment == source.max_frames_per_segment
    assert selected.visibility_post_ms == source.visibility_post_ms
    assert selected.coordinate_transform is source.coordinate_transform


def test_required_curriculum_dataset_rejects_missing_basic_data(
    tmp_path: Path,
) -> None:
    """数据本身缺少基础课程时必须阻断，不能伪装成参数门禁失败。"""

    long_only = SegmentTrainingDataset(
        (
            _record(
                tmp_path,
                key="long",
                dimension="long_sequence",
                category="long_sequence",
            ),
        ),
        split=DataSplit.TRAIN,
        sample_fps=10.0,
        frame_step=1,
        max_frames_per_segment=None,
        visibility_post_ms=100.0,
        coordinate_transform=None,
    )

    with pytest.raises(CurriculumDataUnavailableError, match="basic"):
        require_curriculum_dataset(long_only, CurriculumStage.BASIC)


def test_basic_stage_ignores_full_only_feedback_until_that_frame_is_available(
    tmp_path: Path,
) -> None:
    """FULL 难例不得因 BASIC 无对应帧而制造一个虚假的 Decision 失败。"""

    basic = select_curriculum_dataset(_dataset(tmp_path), CurriculumStage.BASIC)
    weights = _applicable_frame_weights(
        basic,
        {
            ("single", 0): 2.0,
            ("long", 0): 4.0,
        },
    )

    assert weights == {("single", 0): 2.0}
