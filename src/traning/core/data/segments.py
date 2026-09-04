"""从 segment 视频与 item 划分清单构建惰性 typed 训练数据集。

该模块是旧视频/标注读取算法与 V2 领域契约之间的唯一适配边界。下游只会
得到 :class:`TrainingSample`，不会接触 ``dict[str, Any]`` 样本。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import overload

import cv2
from package import AffineOsuVideoTransform
from package.dataset_split import (
    DATASET_SPLIT_SCHEMA_VERSION,
    DatasetSplitManifest,
    load_split_manifest,
)

from traning.conf import DataLoaderConfig, V2Config
from traning.state import (
    DataQualityIssue,
    DataQualityReport,
    DataQualitySeverity,
    DataSplit,
    GroundTruthObject,
    ObjectType,
    Point2D,
    TrainingSample,
)
from traning.core.data.coordinates import FrameCoordinateTransform
from traning.lib.data.annotation import HitObjectAnnotation, visible_hit_objects
from traning.lib.data.discovery import discover_segments
from traning.lib.data.models import FrameReference, SegmentRecord
from traning.lib.data.sampling import build_frame_references
from traning.lib.data.video_reader import VideoReader


_CONCRETE_SPLITS = (
    DataSplit.TRAIN,
    DataSplit.VALIDATION,
    DataSplit.TEST,
)
_DATASET_IDENTITY_VERSION = 1
_SPINNER_CENTER = Point2D(256.0, 192.0)


@dataclass(frozen=True, slots=True)
class DatasetFrameLocation:
    """一个 sequence/frame 在具体 split 数据集中的稳定位置。"""

    sequence_id: str
    frame_index: int
    dataset_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sequence_id, str)
            or not self.sequence_id
            or self.sequence_id != self.sequence_id.strip()
        ):
            raise ValueError("sequence_id 必须非空且无首尾空格")
        for field_name, value in (
            ("frame_index", self.frame_index),
            ("dataset_index", self.dataset_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} 必须是整数")
            if value < 0:
                raise ValueError(f"{field_name} 不得为负数")


class SegmentTrainingDataset(Sequence[TrainingSample]):
    """一个具体 split 的随机访问数据集；构造时不解码任何图像。"""

    def __init__(
        self,
        records: tuple[SegmentRecord, ...],
        *,
        split: DataSplit,
        sample_fps: float,
        frame_step: int,
        max_frames_per_segment: int | None,
        visibility_post_ms: float,
        coordinate_transform: FrameCoordinateTransform | None,
    ) -> None:
        if split not in _CONCRETE_SPLITS:
            raise ValueError("SegmentTrainingDataset 必须属于具体 DataSplit")
        self.records = records
        self.split = split
        self.sample_fps = sample_fps
        self.frame_step = frame_step
        self.max_frames_per_segment = max_frames_per_segment
        self.visibility_post_ms = visibility_post_ms
        self.coordinate_transform = coordinate_transform
        self.references: tuple[FrameReference, ...] = build_frame_references(
            records,
            sample_fps=sample_fps,
            frame_step=frame_step,
            max_frames_per_segment=max_frames_per_segment,
        )
        self._reader: VideoReader | None = None
        self._sequence_indices = self._build_sequence_indices()
        self._frame_locations = self._build_frame_locations()

    def _build_sequence_indices(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """直接按 FrameReference 的 record_index 建立序列索引，不解析 sample_id。"""

        grouped: list[list[int]] = [[] for _ in self.records]
        for dataset_index, reference in enumerate(self.references):
            grouped[reference.record_index].append(dataset_index)
        result = tuple(
            (record.key, tuple(grouped[record_index]))
            for record_index, record in enumerate(self.records)
        )
        sequence_ids = tuple(sequence_id for sequence_id, _ in result)
        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError("同一 split 的 segment sequence_id 不得重复")
        return result

    def _build_frame_locations(
        self,
    ) -> dict[tuple[str, int], DatasetFrameLocation]:
        """从 typed reference 建立 O(1) 帧定位表，不解析 sample_id 文本。"""

        locations: dict[tuple[str, int], DatasetFrameLocation] = {}
        for sequence_id, indices in self._sequence_indices:
            for dataset_index in indices:
                frame_index = self.references[dataset_index].frame_index
                key = (sequence_id, frame_index)
                if key in locations:
                    raise ValueError("同一 sequence 不得重复采样 frame_index")
                locations[key] = DatasetFrameLocation(
                    sequence_id=sequence_id,
                    frame_index=frame_index,
                    dataset_index=dataset_index,
                )
        return locations

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        """按发现顺序返回稳定 segment 身份。"""

        return tuple(sequence_id for sequence_id, _ in self._sequence_indices)

    @property
    def transform_fingerprint(self) -> str | None:
        """返回所有样本共享的坐标变换指纹；阻断配置下为 ``None``。"""

        if self.coordinate_transform is None:
            return None
        return self.coordinate_transform.transform_fingerprint

    def __len__(self) -> int:
        return len(self.references)

    @overload
    def __getitem__(self, index: int) -> TrainingSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TrainingSample, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> TrainingSample | tuple[TrainingSample, ...]:
        if isinstance(index, slice):
            return tuple(
                self[item_index] for item_index in range(*index.indices(len(self)))
            )
        reference = self.references[index]
        record = self.records[reference.record_index]
        transform = self.coordinate_transform
        if transform is None:
            raise RuntimeError("数据质量门已阻断：坐标变换未配置")
        frame = self._video_reader().read_frame_at(
            record.video_path,
            reference.timestamp_ms,
        )
        height, width, channels = frame.shape
        if channels != 3:
            raise ValueError(f"视频帧不是 RGB 三通道: {record.video_path}")
        if (
            width != transform.source_frame_width
            or height != transform.source_frame_height
        ):
            raise ValueError(
                f"视频帧尺寸 {width}x{height} 与坐标标定尺寸不一致: {record.video_path}"
            )
        sample_id = _sample_id(record, reference.frame_index)
        visible = visible_hit_objects(
            record.annotation,
            reference.timestamp_ms,
            visibility_post_ms=self.visibility_post_ms,
        )
        visible_ids = {id(item) for item in visible}
        ground_truth = tuple(
            _ground_truth_object(record, item, object_index)
            for object_index, item in enumerate(record.annotation.hit_objects)
            if id(item) in visible_ids
        )
        # VideoReader 已将 OpenCV 的 BGR 转为 RGB；这里保留连续 uint8 原始字节，
        # 不在数据核心中偷偷缩放、压缩或归一化。
        image_bytes = frame.copy(order="C").tobytes(order="C")
        return TrainingSample(
            sample_id=sample_id,
            split=self.split,
            frame_index=reference.frame_index,
            timestamp_ms=reference.timestamp_ms,
            width=width,
            height=height,
            image_bytes=image_bytes,
            transform_fingerprint=transform.transform_fingerprint,
            candidates=(),
            ground_truth_objects=ground_truth,
            selected_candidate_id=None,
        )

    def _video_reader(self) -> VideoReader:
        """为当前进程惰性创建有限句柄的 LRU 视频读取器。"""

        if self._reader is None:
            self._reader = VideoReader()
        return self._reader

    def sequence(self, sequence_id: str) -> TrainingSequenceDataset:
        """返回一个不复制样本的 typed 因果序列视图。"""

        for current_id, indices in self._sequence_indices:
            if current_id == sequence_id:
                return TrainingSequenceDataset(
                    sequence_id=current_id,
                    split=self.split,
                    source=self,
                    indices=indices,
                )
        raise KeyError(f"未知 sequence_id: {sequence_id}")

    def resolve_sequence_frame(
        self,
        sequence_id: str,
        frame_index: int,
    ) -> DatasetFrameLocation:
        """精确解析 canonical sequence/frame，禁止猜测帧级 sample_id 格式。"""

        if (
            not isinstance(sequence_id, str)
            or not sequence_id
            or sequence_id != sequence_id.strip()
        ):
            raise ValueError("sequence_id 必须非空且无首尾空格")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("frame_index 必须是整数")
        if frame_index < 0:
            raise ValueError("frame_index 不得为负数")
        try:
            return self._frame_locations[(sequence_id, frame_index)]
        except KeyError as error:
            raise KeyError(
                f"未知 sequence/frame: {sequence_id!r}/{frame_index}"
            ) from error

    def iter_sequences(self) -> Iterator[TrainingSequenceDataset]:
        """按稳定发现顺序惰性遍历 segment 序列。"""

        for sequence_id, indices in self._sequence_indices:
            yield TrainingSequenceDataset(
                sequence_id=sequence_id,
                split=self.split,
                source=self,
                indices=indices,
            )

    def close(self) -> None:
        """立即释放当前进程持有的视频句柄。"""

        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __getstate__(self) -> dict[str, object]:
        """DataLoader worker 序列化时不传递 OpenCV 句柄。"""

        state = dict(self.__dict__)
        state["_reader"] = None
        return state

    def __del__(self) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class TrainingSequenceDataset(Sequence[TrainingSample]):
    """一个 segment 的惰性 typed 因果序列视图。"""

    sequence_id: str
    split: DataSplit
    source: SegmentTrainingDataset = field(repr=False, compare=False)
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.sequence_id or self.sequence_id != self.sequence_id.strip():
            raise ValueError("sequence_id 必须非空且无首尾空格")
        if self.split is not self.source.split:
            raise ValueError("sequence split 必须与源数据集一致")
        if any(index < 0 or index >= len(self.source) for index in self.indices):
            raise ValueError("sequence indices 超出源数据集边界")

    def __len__(self) -> int:
        return len(self.indices)

    @overload
    def __getitem__(self, index: int) -> TrainingSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TrainingSample, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> TrainingSample | tuple[TrainingSample, ...]:
        if isinstance(index, slice):
            return tuple(self.source[item] for item in self.indices[index])
        return self.source[self.indices[index]]


class CombinedTrainingDataset(Sequence[TrainingSample]):
    """三个具体 split 的只读拼接视图，用于 ``DataSplit.ALL``。"""

    def __init__(self, datasets: tuple[SegmentTrainingDataset, ...]) -> None:
        if tuple(dataset.split for dataset in datasets) != _CONCRETE_SPLITS:
            raise ValueError("CombinedTrainingDataset 必须按固定 split 顺序构造")
        self.datasets = datasets
        self._locations = tuple(
            (dataset_index, local_index)
            for dataset_index, dataset in enumerate(datasets)
            for local_index in range(len(dataset))
        )

    def __len__(self) -> int:
        return len(self._locations)

    @overload
    def __getitem__(self, index: int) -> TrainingSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TrainingSample, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> TrainingSample | tuple[TrainingSample, ...]:
        if isinstance(index, slice):
            return tuple(
                self[item_index] for item_index in range(*index.indices(len(self)))
            )
        dataset_index, local_index = self._locations[index]
        return self.datasets[dataset_index][local_index]

    def iter_sequences(self) -> Iterator[TrainingSequenceDataset]:
        """依次遍历 train、validation、test 内的全部因果序列。"""

        for dataset in self.datasets:
            yield from dataset.iter_sequences()


@dataclass(frozen=True, slots=True)
class TrainingDatasetBundle:
    """生产数据入口一次返回的数据集、质量、身份与坐标绑定。"""

    datasets: tuple[tuple[DataSplit, SegmentTrainingDataset], ...]
    quality_report: DataQualityReport
    dataset_identity: str
    transform_fingerprint: str | None
    loader: DataLoaderConfig
    coordinate_transform: FrameCoordinateTransform | None = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if tuple(split for split, _ in self.datasets) != _CONCRETE_SPLITS:
            raise ValueError("datasets 必须按 train、validation、test 顺序完整提供")
        if any(dataset.split is not split for split, dataset in self.datasets):
            raise ValueError("dataset.split 与 bundle 注册键不一致")
        if not isinstance(self.quality_report, DataQualityReport):
            raise TypeError("quality_report 必须是 DataQualityReport")
        if (
            not self.dataset_identity.startswith("dataset-")
            or len(self.dataset_identity) != len("dataset-") + 64
        ):
            raise ValueError("dataset_identity 必须是 dataset- 前缀的 SHA-256")
        if not isinstance(self.loader, DataLoaderConfig):
            raise TypeError("loader 必须是 DataLoaderConfig")
        fingerprints = {dataset.transform_fingerprint for _, dataset in self.datasets}
        if fingerprints != {self.transform_fingerprint}:
            raise ValueError("bundle 内所有 split 的 transform_fingerprint 必须一致")
        if self.coordinate_transform is None:
            if self.transform_fingerprint is not None:
                raise ValueError("缺少 coordinate_transform 时不得声明指纹")
        elif (
            self.transform_fingerprint
            != self.coordinate_transform.transform_fingerprint
        ):
            raise ValueError("coordinate_transform 与 bundle 指纹不一致")

    def dataset(
        self,
        split: DataSplit,
    ) -> SegmentTrainingDataset | CombinedTrainingDataset:
        """按唯一 DataSplit 返回 typed 数据集；ALL 返回惰性拼接视图。"""

        if split is DataSplit.ALL:
            return self.all
        for registered_split, dataset in self.datasets:
            if registered_split is split:
                return dataset
        raise ValueError("split 必须是 DataSplit")

    @property
    def train(self) -> SegmentTrainingDataset:
        """返回训练 split。"""

        return self._concrete_dataset(DataSplit.TRAIN)

    @property
    def validation(self) -> SegmentTrainingDataset:
        """返回验证 split。"""

        return self._concrete_dataset(DataSplit.VALIDATION)

    @property
    def test(self) -> SegmentTrainingDataset:
        """返回测试 split。"""

        return self._concrete_dataset(DataSplit.TEST)

    @property
    def all(self) -> CombinedTrainingDataset:
        """返回固定 split 顺序的惰性拼接视图。"""

        return CombinedTrainingDataset(tuple(dataset for _, dataset in self.datasets))

    def _concrete_dataset(self, split: DataSplit) -> SegmentTrainingDataset:
        for registered_split, dataset in self.datasets:
            if registered_split is split:
                return dataset
        raise RuntimeError(f"bundle 缺少 {split.value} 数据集")


def build_training_datasets(config: V2Config) -> TrainingDatasetBundle:
    """发现生产 segment 并返回 typed bundle；固定阻断写入 report 而非抛出。

    构造阶段会读取标注、清单、文件摘要和视频头，但不会解码训练帧。调用方必须
    在访问 dataset 前对 ``quality_report`` 执行 canonical ``require_quality``。
    """

    if not isinstance(config, V2Config):
        raise TypeError("config 必须是 V2Config")
    issues: list[DataQualityIssue] = []
    transform = _coordinate_transform(config, issues)
    manifest = _split_manifest(config, issues)
    discovery = discover_segments(config.data.dataset_root)
    issues.extend(
        _issue(
            "segment_discovery_error",
            str(discovery_issue.message),
            blocks_training=True,
            details=(("path", str(discovery_issue.path)),),
        )
        for discovery_issue in discovery.issues
    )

    grouped: dict[DataSplit, list[SegmentRecord]] = {
        split: [] for split in _CONCRETE_SPLITS
    }
    if manifest is not None:
        _assign_records(discovery.records, manifest, grouped, issues)
    selected = {
        split: tuple(records[: config.data.max_segments_per_split])
        if config.data.max_segments_per_split is not None
        else tuple(records)
        for split, records in grouped.items()
    }
    _validate_records(selected, config, issues)

    datasets = tuple(
        (
            split,
            SegmentTrainingDataset(
                selected[split],
                split=split,
                sample_fps=config.data.sample_fps,
                frame_step=config.data.frame_step,
                max_frames_per_segment=config.data.max_frames_per_segment,
                visibility_post_ms=config.data.visibility_post_ms,
                coordinate_transform=transform,
            ),
        )
        for split in _CONCRETE_SPLITS
    )
    _append_split_quality(datasets, issues)
    dataset_identity = _dataset_identity(config, selected, transform)
    report = DataQualityReport(issues=tuple(sorted(issues, key=_issue_sort_key)))
    return TrainingDatasetBundle(
        datasets=datasets,
        quality_report=report,
        dataset_identity=dataset_identity,
        transform_fingerprint=(
            None if transform is None else transform.transform_fingerprint
        ),
        loader=config.data.loader,
        coordinate_transform=transform,
    )


def _coordinate_transform(
    config: V2Config,
    issues: list[DataQualityIssue],
) -> FrameCoordinateTransform | None:
    coordinate_config = config.coordinates
    if coordinate_config.affine_matrix is None:
        issues.append(
            _issue(
                "coordinate_transform_unconfigured",
                "训练数据缺少 coordinates.affine_matrix",
                blocks_training=True,
            )
        )
        return None
    try:
        return FrameCoordinateTransform(
            source_frame_width=coordinate_config.source_width,
            source_frame_height=coordinate_config.source_height,
            transform_identity=coordinate_config.transform_identity,
            transform=AffineOsuVideoTransform(coordinate_config.affine_matrix),
        )
    except (TypeError, ValueError) as error:
        issues.append(
            _issue(
                "coordinate_transform_invalid",
                f"训练坐标变换无效: {error}",
                blocks_training=True,
            )
        )
        return None


def _split_manifest(
    config: V2Config,
    issues: list[DataQualityIssue],
) -> DatasetSplitManifest | None:
    try:
        manifest = load_split_manifest(config.data.split_manifest)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        issues.append(
            _issue(
                "split_manifest_invalid",
                f"item split 清单无法读取: {error}",
                blocks_training=True,
                details=(("path", str(config.data.split_manifest)),),
            )
        )
        return None
    if manifest is None:
        issues.append(
            _issue(
                "split_manifest_missing",
                "item split 清单不存在",
                blocks_training=True,
                details=(("path", str(config.data.split_manifest)),),
            )
        )
        return None
    if manifest.schema_version != DATASET_SPLIT_SCHEMA_VERSION:
        issues.append(
            _issue(
                "split_manifest_schema_mismatch",
                "item split 清单 schema_version 不受支持",
                blocks_training=True,
                details=(
                    ("actual", manifest.schema_version),
                    ("expected", DATASET_SPLIT_SCHEMA_VERSION),
                ),
            )
        )
    if manifest.unit != "item":
        issues.append(
            _issue(
                "split_manifest_unit_mismatch",
                "item split 清单必须以 item 为划分单位",
                blocks_training=True,
                details=(("unit", manifest.unit),),
            )
        )
    if manifest.seed != config.data.seed:
        issues.append(
            _issue(
                "split_manifest_seed_mismatch",
                "item split 清单 seed 与训练配置不一致",
                blocks_training=True,
                details=(
                    ("manifest_seed", manifest.seed),
                    ("config_seed", config.data.seed),
                ),
            )
        )
    return manifest


def _assign_records(
    records: tuple[SegmentRecord, ...],
    manifest: DatasetSplitManifest,
    grouped: dict[DataSplit, list[SegmentRecord]],
    issues: list[DataQualityIssue],
) -> None:
    """只依据冻结 item 归属分组，禁止同一谱面按 segment 再切分。"""

    discovered_items: set[str] = set()
    for record in records:
        discovered_items.add(record.item_name)
        manifest_item = manifest.items.get(record.item_name)
        if manifest_item is None:
            issues.append(
                _issue(
                    "item_without_split",
                    "发现的数据 item 未在 split 清单中冻结归属",
                    blocks_training=True,
                    sample_id=record.key,
                    details=(("item_name", record.item_name),),
                )
            )
            continue
        try:
            split = DataSplit(_enum_value(manifest_item.split))
        except ValueError:
            issues.append(
                _issue(
                    "item_split_invalid",
                    "item 的 split 值无效",
                    blocks_training=True,
                    sample_id=record.key,
                    details=(("split", _enum_value(manifest_item.split)),),
                )
            )
            continue
        if split is DataSplit.ALL:
            issues.append(
                _issue(
                    "item_split_all_forbidden",
                    "item 不能归属聚合 split=all",
                    blocks_training=True,
                    sample_id=record.key,
                )
            )
            continue
        grouped[split].append(record)

    for item_name in sorted(set(manifest.items) - discovered_items):
        issues.append(
            _issue(
                "manifest_item_not_found",
                "split 清单中的 item 当前没有 segment",
                blocks_training=False,
                severity=DataQualitySeverity.WARNING,
                details=(("item_name", item_name),),
            )
        )


def _validate_records(
    selected: dict[DataSplit, tuple[SegmentRecord, ...]],
    config: V2Config,
    issues: list[DataQualityIssue],
) -> None:
    seen_keys: set[str] = set()
    for split in _CONCRETE_SPLITS:
        for record in selected[split]:
            if record.key in seen_keys:
                issues.append(
                    _issue(
                        "duplicate_segment_key",
                        "segment key 在数据集中重复",
                        blocks_training=True,
                        sample_id=record.key,
                    )
                )
            seen_keys.add(record.key)
            _validate_annotation(record, issues)
            _validate_video_header(record, config, issues)


def _validate_annotation(
    record: SegmentRecord,
    issues: list[DataQualityIssue],
) -> None:
    object_ids: list[str] = []
    for object_index, item in enumerate(record.annotation.hit_objects):
        try:
            target = _ground_truth_object(record, item, object_index)
        except (TypeError, ValueError) as error:
            issues.append(
                _issue(
                    "ground_truth_invalid",
                    f"标注无法转换为 GroundTruthObject: {error}",
                    blocks_training=True,
                    sample_id=record.key,
                    details=(("object_index", object_index),),
                )
            )
            continue
        object_ids.append(target.object_id)
    if len(object_ids) != len(set(object_ids)):
        issues.append(
            _issue(
                "duplicate_ground_truth_id",
                "同一 segment 的 ground truth object_id 重复",
                blocks_training=True,
                sample_id=record.key,
            )
        )


def _validate_video_header(
    record: SegmentRecord,
    config: V2Config,
    issues: list[DataQualityIssue],
) -> None:
    capture = cv2.VideoCapture(str(record.video_path))
    try:
        if not capture.isOpened():
            issues.append(
                _issue(
                    "video_unreadable",
                    "视频无法打开",
                    blocks_training=True,
                    sample_id=record.key,
                    details=(("path", str(record.video_path)),),
                )
            )
            return
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if (width, height) != (
            config.coordinates.source_width,
            config.coordinates.source_height,
        ):
            issues.append(
                _issue(
                    "video_coordinate_size_mismatch",
                    "视频尺寸与坐标标定尺寸不一致",
                    blocks_training=True,
                    sample_id=record.key,
                    details=(
                        ("video_width", width),
                        ("video_height", height),
                        ("coordinate_width", config.coordinates.source_width),
                        ("coordinate_height", config.coordinates.source_height),
                    ),
                )
            )
        if frame_count < 1:
            issues.append(
                _issue(
                    "video_empty",
                    "视频头声明的帧数为空",
                    blocks_training=True,
                    sample_id=record.key,
                )
            )
    finally:
        capture.release()


def _append_split_quality(
    datasets: tuple[tuple[DataSplit, SegmentTrainingDataset], ...],
    issues: list[DataQualityIssue],
) -> None:
    counts = {split: len(dataset) for split, dataset in datasets}
    if counts[DataSplit.TRAIN] == 0:
        issues.append(
            _issue(
                "missing_training_split",
                "训练切分没有样本",
                blocks_training=True,
            )
        )
    for split in (DataSplit.VALIDATION, DataSplit.TEST):
        if counts[split] == 0:
            issues.append(
                _issue(
                    "missing_evaluation_split",
                    f"{split.value} 切分没有样本",
                    blocks_training=False,
                    severity=DataQualitySeverity.WARNING,
                    details=(("split", split.value),),
                )
            )


def _ground_truth_object(
    record: SegmentRecord,
    item: HitObjectAnnotation,
    object_index: int,
) -> GroundTruthObject:
    """把 permissive 标注模型收口成严格 GroundTruthObject。"""

    item_type = item.type.strip().lower()
    object_id = _object_id(record, item, object_index)
    if item_type in {"circle", "ring", "point"}:
        if item.x is None or item.y is None:
            raise ValueError("circle 缺少 x/y")
        return GroundTruthObject(
            object_id=object_id,
            object_type=ObjectType.RING,
            position=Point2D(float(item.x), float(item.y)),
            start_time_ms=float(item.start_ms),
            end_time_ms=float(item.end_ms),
            score=1.0,
            radius_osu=float(record.annotation.difficulty.circle_radius_osu_pixels),
        )
    if item_type == "slider":
        path = tuple(Point2D(float(x), float(y)) for x, y in item.path)
        if len(path) < 2:
            raise ValueError("slider path 少于两个点")
        return GroundTruthObject(
            object_id=object_id,
            object_type=ObjectType.SLIDER,
            position=path[0],
            start_time_ms=float(item.start_ms),
            end_time_ms=float(item.end_ms),
            score=1.0,
            path=path,
        )
    if item_type == "spinner":
        return GroundTruthObject(
            object_id=object_id,
            object_type=ObjectType.SPINNER,
            position=_SPINNER_CENTER,
            start_time_ms=float(item.start_ms),
            end_time_ms=float(item.end_ms),
            score=1.0,
        )
    raise ValueError(f"不支持的 hit object type: {item.type!r}")


def _object_id(
    record: SegmentRecord,
    item: HitObjectAnnotation,
    object_index: int,
) -> str:
    if item.source_index is not None:
        # 同一谱面目标出现在重叠 segment 时仍共享稳定身份。
        return f"{record.item_name}:source-{item.source_index}"
    return f"{record.key}:object-{object_index:04d}"


def _sample_id(record: SegmentRecord, frame_index: int) -> str:
    return f"{record.key}:frame-{frame_index:06d}"


def _dataset_identity(
    config: V2Config,
    selected: dict[DataSplit, tuple[SegmentRecord, ...]],
    transform: FrameCoordinateTransform | None,
) -> str:
    """对实际消费的清单、标注和视频内容生成稳定 SHA-256 身份。"""

    hasher = hashlib.sha256()
    header = {
        "identity_version": _DATASET_IDENTITY_VERSION,
        "seed": config.data.seed,
        "sample_fps": config.data.sample_fps,
        "frame_step": config.data.frame_step,
        "max_segments_per_split": config.data.max_segments_per_split,
        "max_frames_per_segment": config.data.max_frames_per_segment,
        "visibility_post_ms": config.data.visibility_post_ms,
        "transform_fingerprint": (
            None if transform is None else transform.transform_fingerprint
        ),
        "manifest_sha256": _file_digest(config.data.split_manifest),
    }
    hasher.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for split in _CONCRETE_SPLITS:
        for record in selected[split]:
            relative_directory = _relative_path(
                record.directory, config.data.dataset_root
            )
            row = {
                "split": split.value,
                "record_key": record.key,
                "directory": relative_directory,
                "annotation_sha256": _file_digest(record.annotation_path),
                "video_sha256": _file_digest(record.video_path),
            }
            hasher.update(b"\n")
            hasher.update(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return f"dataset-{hasher.hexdigest()}"


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError as error:
        # 质量报告会给出可操作阻断；身份仍必须可复现且不能因异常直接缺失。
        return f"unreadable:{type(error).__name__}:{path}"
    return hasher.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _issue(
    code: str,
    message: str,
    *,
    blocks_training: bool,
    severity: DataQualitySeverity = DataQualitySeverity.ERROR,
    sample_id: str | None = None,
    details: tuple[tuple[str, str | int | float | bool | None], ...] = (),
) -> DataQualityIssue:
    return DataQualityIssue(
        code=code,
        severity=severity,
        blocks_training=blocks_training,
        sample_id=sample_id,
        message=message,
        details=details,
    )


def _issue_sort_key(issue: DataQualityIssue) -> tuple[str, str, str]:
    return issue.code, issue.sample_id or "", issue.message


def _enum_value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


__all__ = (
    "CombinedTrainingDataset",
    "DatasetFrameLocation",
    "SegmentTrainingDataset",
    "TrainingDatasetBundle",
    "TrainingSequenceDataset",
    "build_training_datasets",
)
