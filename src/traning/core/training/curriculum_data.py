"""按 canonical curriculum 规格选择累计难度的 segment 数据视图。"""

from __future__ import annotations

from dataclasses import dataclass

from package import CurriculumStage

from traning.core.data import SegmentTrainingDataset


@dataclass(frozen=True, slots=True)
class CurriculumDataSpec:
    """一个课程阶段允许的数据维度与类别集合。"""

    stage: CurriculumStage
    dimensions: frozenset[str] | None
    categories: frozenset[str] | None

    def accepts(self, *, dimension: str, category: str) -> bool:
        """判断 segment 是否属于当前累计课程范围。"""

        dimension_allowed = self.dimensions is None or dimension in self.dimensions
        category_allowed = self.categories is None or category in self.categories
        return dimension_allowed and category_allowed


CURRICULUM_DATA_REGISTRY: tuple[CurriculumDataSpec, ...] = (
    CurriculumDataSpec(
        CurriculumStage.BASIC,
        frozenset({"atomic"}),
        frozenset({"single_point", "slider"}),
    ),
    CurriculumDataSpec(
        CurriculumStage.MULTI_OBJECT,
        frozenset({"atomic"}),
        frozenset({"single_point", "slider", "multi_point"}),
    ),
    CurriculumDataSpec(
        CurriculumStage.COMPLEX,
        frozenset({"atomic"}),
        None,
    ),
    CurriculumDataSpec(CurriculumStage.FULL, None, None),
)
"""累计课程数据规格；后续阶段只能增加样本，不能回退到更简单子集。"""


class CurriculumDataUnavailableError(RuntimeError):
    """课程必要 split 没有样本，不能靠更换参数修复。"""


def select_curriculum_dataset(
    dataset: SegmentTrainingDataset,
    stage: CurriculumStage,
) -> SegmentTrainingDataset:
    """返回保留原 split/采样/坐标契约的课程数据集。"""

    if not isinstance(dataset, SegmentTrainingDataset):
        raise TypeError("dataset 必须是 SegmentTrainingDataset")
    if not isinstance(stage, CurriculumStage):
        raise TypeError("stage 必须是 package.CurriculumStage")
    registry = {spec.stage: spec for spec in CURRICULUM_DATA_REGISTRY}
    spec = registry[stage]
    records = tuple(
        record
        for record in dataset.records
        if spec.accepts(
            dimension=record.dataset_dimension,
            category=record.category,
        )
    )
    return SegmentTrainingDataset(
        records,
        split=dataset.split,
        sample_fps=dataset.sample_fps,
        frame_step=dataset.frame_step,
        max_frames_per_segment=dataset.max_frames_per_segment,
        visibility_post_ms=dataset.visibility_post_ms,
        coordinate_transform=dataset.coordinate_transform,
    )


def require_curriculum_dataset(
    dataset: SegmentTrainingDataset,
    stage: CurriculumStage,
) -> SegmentTrainingDataset:
    """选择课程视图，并把空 split 作为不可调参的数据错误阻断。"""

    selected = select_curriculum_dataset(dataset, stage)
    if len(selected) == 0:
        raise CurriculumDataUnavailableError(
            f"{dataset.split.value} split 在 {stage.value} curriculum 没有训练帧"
        )
    return selected


__all__ = (
    "CURRICULUM_DATA_REGISTRY",
    "CurriculumDataSpec",
    "CurriculumDataUnavailableError",
    "require_curriculum_dataset",
    "select_curriculum_dataset",
)
