"""把不可变 telemetry 快照纯投影为 Rich/Qt 视图模型。

本模块故意不导入具体 GUI 库，也不持有 store/reporter。调用方取得
``DashboardSnapshot`` 后才调用 renderer，因此训练线程与展示线程之间没有可变
live state，也不会在展示层重新判定质量、评分或错误归因。
"""

from __future__ import annotations

from io import BytesIO
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeAlias

from PIL import Image, ImageDraw

from traning.contracts import Point2D, RuntimeFrame
from traning.data.coordinates import (
    FrameCoordinateTransform,
    FramePixelPoint,
    FrameProjectedPoint,
    OsuPoint,
)
from traning.evaluation.attribution import (
    EvaluationTag,
    PrimaryError,
    SequenceEvaluationEvent,
)
from traning.evaluation.sequence import FrameSequenceScore, TargetObject
from traning.infrastructure import atomic_write_bytes
from traning.telemetry.reporter import DashboardSnapshot


MetricNumber: TypeAlias = int | float
"""仪表盘允许展示的不可变数值类型。"""

MetricExtractor: TypeAlias = Callable[[DashboardSnapshot], MetricNumber | None]
"""从唯一 telemetry snapshot 读取一个值的纯函数。"""


class DashboardSection(str, Enum):
    """Rich 分区和 Qt 分组共用的稳定领域顺序。"""

    TRAINING = "training"
    PERCEPTION = "perception"
    TRACKING = "tracking"
    OUTCOME = "outcome"
    DECISION = "decision"
    RESOURCES = "resources"


class DashboardMetric(str, Enum):
    """Phase 10 必须可视化的完整指标集合。"""

    STEP = "step"
    LOSS = "loss"
    SCORE = "score"
    PERCEPTION_RECALL = "perception_recall"
    TRACKING_ID_SWITCHES = "tracking_id_switches"
    OUTCOME_NLL = "outcome_nll"
    OUTCOME_BRIER = "outcome_brier"
    OUTCOME_ECE = "outcome_ece"
    EXPECTED_SCORE_ERROR = "expected_score_error"
    DECISION_UTILITY = "decision_utility"
    WAIT_CLICK_RATIO = "wait_click_ratio"
    THROUGHPUT = "throughput"
    GPU_UTILIZATION = "gpu_utilization"
    VRAM_USED_MB = "vram_used_mb"
    VRAM_TOTAL_MB = "vram_total_mb"


class QtMetricColumn(str, Enum):
    """Qt 指标表的强类型列标识。"""

    SECTION = "section"
    METRIC = "metric"
    VALUE = "value"
    UNIT = "unit"


class QtEvaluationColumn(str, Enum):
    """Qt evaluation 表的强类型列标识。"""

    EVENT_ID = "event_id"
    SAMPLE_ID = "sample_id"
    FRAME_INDEX = "frame_index"
    PASSED = "passed"
    PRIMARY_ERROR = "primary_error"
    ERROR_TAGS = "error_tags"
    TARGET_ID = "target_id"
    CLICK_INDEX = "click_index"


@dataclass(frozen=True, slots=True)
class _MetricSpec:
    """集中定义取值、标签和格式，避免 Rich/Qt 各自解释指标。"""

    metric: DashboardMetric
    section: DashboardSection
    label: str
    unit: str
    precision: int
    extract: MetricExtractor


@dataclass(frozen=True, slots=True)
class DashboardMetricRow:
    """两个 renderer 共用的不可变指标行。"""

    rank: int
    metric: DashboardMetric
    section: DashboardSection
    label: str
    value: MetricNumber | None
    display_value: str
    unit: str

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank 必须是整数")
        if self.rank < 0:
            raise ValueError("rank 不得为负数")
        if not isinstance(self.metric, DashboardMetric):
            raise TypeError("metric 必须是 DashboardMetric")
        if not isinstance(self.section, DashboardSection):
            raise TypeError("section 必须是 DashboardSection")
        if not self.label or self.label != self.label.strip():
            raise ValueError("label 必须非空且无首尾空格")
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise TypeError("value 必须是数值或 None")
            if not math.isfinite(float(self.value)):
                raise ValueError("value 必须是有限数值")
        if not self.display_value:
            raise ValueError("display_value 不得为空")
        if not isinstance(self.unit, str):
            raise TypeError("unit 必须是字符串")


@dataclass(frozen=True, slots=True)
class DashboardEvaluationRow:
    """直接持有 canonical event，确保 UI 不复制或重新归因。"""

    event: SequenceEvaluationEvent

    def __post_init__(self) -> None:
        if not isinstance(self.event, SequenceEvaluationEvent):
            raise TypeError("event 必须是 SequenceEvaluationEvent")

    @property
    def event_id(self) -> str:
        """返回 canonical event identity。"""

        return self.event.event_id

    @property
    def sample_id(self) -> str:
        """返回 scorer 写入的样本标识。"""

        return self.event.sample_id

    @property
    def frame_index(self) -> int:
        """返回 scorer 写入的帧序号。"""

        return self.event.frame_index

    @property
    def passed(self) -> bool:
        """原样展示 canonical pass，不在 UI 重新计算。"""

        return self.event.passed

    @property
    def primary_error(self) -> PrimaryError:
        """原样展示 canonical primary_error，不读取 error tag 猜测。"""

        return self.event.primary_error

    @property
    def error_tags(self) -> tuple[EvaluationTag, ...]:
        """返回 canonical 次级标签。"""

        return self.event.error_tags

    @property
    def target_id(self) -> str | None:
        """返回 scorer 绑定的目标标识。"""

        return self.event.target_id

    @property
    def click_index(self) -> int | None:
        """返回 scorer 绑定的点击序号。"""

        return self.event.click_index


@dataclass(frozen=True, slots=True)
class GalleryTargetOverlay:
    """gallery 在原帧上绘制的强类型目标中心与 slider 路径。"""

    target_id: str
    head: FramePixelPoint
    path: tuple[FrameProjectedPoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str):
            raise TypeError("target_id 必须是字符串")
        if not self.target_id or self.target_id != self.target_id.strip():
            raise ValueError("target_id 必须非空且无首尾空格")
        if not isinstance(self.head, FramePixelPoint):
            raise TypeError("head 必须是 FramePixelPoint")
        if not isinstance(self.path, tuple) or any(
            not isinstance(point, FrameProjectedPoint) for point in self.path
        ):
            raise TypeError("path 必须是 FrameProjectedPoint 元组")
        points = (self.head, *self.path)
        if any(
            point.transform_fingerprint != self.head.transform_fingerprint
            or point.source_frame_width != self.head.source_frame_width
            or point.source_frame_height != self.head.source_frame_height
            for point in points
        ):
            raise ValueError("gallery overlay 所有点必须共享帧尺寸和坐标指纹")


def project_gallery_target_overlays(
    targets: tuple[TargetObject, ...],
    coordinate_transform: FrameCoordinateTransform,
) -> tuple[GalleryTargetOverlay, ...]:
    """用与训练和评分同一的变换投影 gallery 目标，不在 renderer 加偏移。"""

    if not isinstance(targets, tuple) or any(
        not isinstance(target, TargetObject) for target in targets
    ):
        raise TypeError("targets 必须是 TargetObject tuple")
    if not isinstance(coordinate_transform, FrameCoordinateTransform):
        raise TypeError("coordinate_transform 必须是 FrameCoordinateTransform")

    overlays: list[GalleryTargetOverlay] = []
    ordered_targets = sorted(
        targets,
        key=lambda target: (
            target.start_ms,
            target.source_index if target.source_index is not None else 10**12,
            target.target_id,
        ),
    )
    for target in ordered_targets:
        if target.x is not None and target.y is not None:
            head = OsuPoint(target.x, target.y)
        else:
            head = OsuPoint(*target.path[0])
        frame_head = coordinate_transform.target_to_gallery_overlay(
            head,
            source_frame_width=coordinate_transform.source_frame_width,
            source_frame_height=coordinate_transform.source_frame_height,
        )
        frame_path = tuple(
            coordinate_transform.ground_truth_geometry_to_frame(
                Point2D(x, y),
                source_frame_width=coordinate_transform.source_frame_width,
                source_frame_height=coordinate_transform.source_frame_height,
            )
            for x, y in target.path
        )
        overlays.append(
            GalleryTargetOverlay(
                target_id=target.target_id,
                head=frame_head,
                path=frame_path,
            )
        )
    return tuple(overlays)


@dataclass(frozen=True, slots=True)
class GalleryPredictionOverlay:
    """原帧预测点及其 scorer 产生的原始 canonical 事件。"""

    position: FramePixelPoint
    event: SequenceEvaluationEvent

    def __post_init__(self) -> None:
        if not isinstance(self.position, FramePixelPoint):
            raise TypeError("position 必须是 FramePixelPoint")
        if not isinstance(self.event, SequenceEvaluationEvent):
            raise TypeError("event 必须是 SequenceEvaluationEvent")
        if self.event.click_index is None:
            raise ValueError("prediction overlay 必须绑定 click 事件")
        if (
            self.event.coordinate_transform_fingerprint
            != self.position.transform_fingerprint
            or self.event.source_frame_width != self.position.source_frame_width
            or self.event.source_frame_height != self.position.source_frame_height
        ):
            raise ValueError("prediction overlay 的事件与像素点坐标来源不一致")
        if (
            self.event.click_x != self.position.x
            or self.event.click_y != self.position.y
        ):
            raise ValueError("prediction overlay 的事件坐标与原帧点不一致")


@dataclass(frozen=True, slots=True)
class GalleryFrameOverlay:
    """一个原帧的 GT、预测和完整归因事件不可变集合。"""

    frame_index: int
    source_frame_width: int
    source_frame_height: int
    transform_fingerprint: str
    targets: tuple[GalleryTargetOverlay, ...]
    predictions: tuple[GalleryPredictionOverlay, ...]
    events: tuple[SequenceEvaluationEvent, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("gallery frame_index 必须是非负整数")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.source_frame_width, self.source_frame_height)
        ):
            raise ValueError("gallery 原帧尺寸必须是正整数")
        if not isinstance(
            self.transform_fingerprint, str
        ) or not self.transform_fingerprint.startswith("transform-"):
            raise ValueError("gallery 必须携带有效坐标变换指纹")
        collection_specs = (
            ("targets", self.targets, GalleryTargetOverlay),
            ("predictions", self.predictions, GalleryPredictionOverlay),
            ("events", self.events, SequenceEvaluationEvent),
        )
        for name, values, item_type in collection_specs:
            if not isinstance(values, tuple) or any(
                not isinstance(value, item_type) for value in values
            ):
                raise TypeError(f"{name} 含有错误类型")
        frame_points = tuple(
            point for target in self.targets for point in (target.head, *target.path)
        ) + tuple(prediction.position for prediction in self.predictions)
        if any(
            point.source_frame_width != self.source_frame_width
            or point.source_frame_height != self.source_frame_height
            or point.transform_fingerprint != self.transform_fingerprint
            for point in frame_points
        ):
            raise ValueError("gallery 所有坐标必须共享尺寸与变换指纹")
        if any(
            event.coordinate_transform_fingerprint != self.transform_fingerprint
            or event.source_frame_width != self.source_frame_width
            or event.source_frame_height != self.source_frame_height
            for event in self.events
        ):
            raise ValueError("gallery 所有事件必须来自同一 calibrated frame 坐标系")
        if any(event.frame_index != self.frame_index for event in self.events):
            raise ValueError("gallery 事件必须精确属于当前原帧")


def build_gallery_frame_overlay(
    targets: tuple[TargetObject, ...],
    score: FrameSequenceScore,
    events: tuple[SequenceEvaluationEvent, ...],
    coordinate_transform: FrameCoordinateTransform,
    *,
    frame_index: int,
) -> GalleryFrameOverlay:
    """把 scorer 原始事件和同一坐标变换组合成可直接渲染的原帧 overlay。"""

    if not isinstance(score, FrameSequenceScore):
        raise TypeError("score 必须是 FrameSequenceScore")
    if not isinstance(events, tuple) or any(
        not isinstance(event, SequenceEvaluationEvent) for event in events
    ):
        raise TypeError("events 必须是 SequenceEvaluationEvent 元组")
    if not isinstance(coordinate_transform, FrameCoordinateTransform):
        raise TypeError("coordinate_transform 必须是 FrameCoordinateTransform")
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
    ):
        raise ValueError("frame_index 必须是非负整数")
    if (
        score.transform_fingerprint != coordinate_transform.transform_fingerprint
        or score.source_frame_width != coordinate_transform.source_frame_width
        or score.source_frame_height != coordinate_transform.source_frame_height
    ):
        raise ValueError("frame score 与 gallery 坐标变换来源不一致")

    frame_events = tuple(event for event in events if event.frame_index == frame_index)
    click_events = tuple(
        sorted(
            (event for event in frame_events if event.click_index is not None),
            key=lambda event: event.click_index,
        )
    )
    click_indices = tuple(event.click_index for event in click_events)
    if len(click_indices) != len(set(click_indices)) or any(
        index is None or index >= len(score.frame_clicks) for index in click_indices
    ):
        raise ValueError("gallery click events 含有重复或越界 click_index")
    if any(
        score.frame_clicks[index].frame_index not in {None, frame_index}
        for index in click_indices
        if index is not None
    ):
        raise ValueError("gallery click event 与 FramePredictedClick 来源帧不一致")
    unresolved_frames = dict(score.unresolved_target_frame_indices)
    expected_unresolved = tuple(
        sorted(
            target_id
            for target_id in score.unresolved_target_ids
            if unresolved_frames.get(target_id, frame_index) == frame_index
        )
    )
    actual_unresolved = tuple(
        sorted(event.target_id for event in frame_events if event.click_index is None)
    )
    if actual_unresolved != expected_unresolved:
        raise ValueError("gallery unresolved events 与 frame score 不一致")

    predictions = tuple(
        GalleryPredictionOverlay(
            position=score.frame_clicks[event.click_index].position,
            event=event,
        )
        for event in click_events
        if event.click_index is not None
    )
    frame_target_ids = {
        event.target_id for event in frame_events if event.target_id is not None
    }
    return GalleryFrameOverlay(
        frame_index=frame_index,
        source_frame_width=score.source_frame_width,
        source_frame_height=score.source_frame_height,
        transform_fingerprint=score.transform_fingerprint,
        targets=project_gallery_target_overlays(
            tuple(
                target
                for target in targets
                if target.frame_index in {None, frame_index}
                or target.target_id in frame_target_ids
            ),
            coordinate_transform,
        ),
        predictions=predictions,
        events=frame_events,
    )


def render_gallery_png(
    frame: RuntimeFrame,
    overlay: GalleryFrameOverlay,
    output_path: Path,
) -> None:
    """在原始 RGB 帧上绘制 GT 与预测点击，并原子发布真实 PNG 文件。"""

    if not isinstance(frame, RuntimeFrame):
        raise TypeError("frame 必须是 RuntimeFrame")
    if not isinstance(overlay, GalleryFrameOverlay):
        raise TypeError("overlay 必须是 GalleryFrameOverlay")
    if not isinstance(output_path, Path):
        raise TypeError("output_path 必须是 pathlib.Path")
    if (frame.width, frame.height) != (
        overlay.source_frame_width,
        overlay.source_frame_height,
    ):
        raise ValueError("RuntimeFrame 尺寸与 gallery overlay 不一致")
    if frame.frame_index != overlay.frame_index:
        raise ValueError("RuntimeFrame.frame_index 与 gallery overlay 不一致")
    expected_size = frame.width * frame.height * 3
    if len(frame.image_bytes) != expected_size:
        raise ValueError("RuntimeFrame.image_bytes 必须是完整 packed RGB 数据")

    image = Image.frombytes("RGB", (frame.width, frame.height), frame.image_bytes)
    draw = ImageDraw.Draw(image)
    for target in overlay.targets:
        if len(target.path) >= 2:
            draw.line(
                tuple((point.x, point.y) for point in target.path),
                fill=(0, 220, 80),
                width=3,
            )
        _draw_cross(draw, target.head.x, target.head.y, color=(0, 220, 80), radius=7)
    for prediction in overlay.predictions:
        color = (40, 150, 255) if prediction.event.passed else (255, 70, 70)
        _draw_cross(
            draw,
            prediction.position.x,
            prediction.position.y,
            color=color,
            radius=5,
        )

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    atomic_write_bytes(output_path, buffer.getvalue())


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    *,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    """用确定性整数像素绘制带外框的十字标记。"""

    center_x = round(x)
    center_y = round(y)
    draw.ellipse(
        (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ),
        outline=color,
        width=2,
    )
    draw.line(
        (center_x - radius, center_y, center_x + radius, center_y),
        fill=color,
        width=1,
    )
    draw.line(
        (center_x, center_y - radius, center_x, center_y + radius),
        fill=color,
        width=1,
    )


@dataclass(frozen=True, slots=True)
class RichMetricSection:
    """Rich 页面中的一个稳定分区。"""

    section: DashboardSection
    title: str
    rows: tuple[DashboardMetricRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.section, DashboardSection):
            raise TypeError("section 必须是 DashboardSection")
        if not self.title or self.title != self.title.strip():
            raise ValueError("title 必须非空且无首尾空格")
        if not isinstance(self.rows, tuple) or any(
            not isinstance(row, DashboardMetricRow) for row in self.rows
        ):
            raise TypeError("rows 必须是 DashboardMetricRow 元组")
        if any(row.section is not self.section for row in self.rows):
            raise ValueError("Rich 分区只能包含本分区指标")
        if tuple(row.rank for row in self.rows) != tuple(
            sorted(row.rank for row in self.rows)
        ):
            raise ValueError("Rich 指标行必须稳定排序")


@dataclass(frozen=True, slots=True)
class RichDashboardModel:
    """不依赖 ``rich`` 包的终端 dashboard 纯 view-model。"""

    schema_version: int
    run_id: str
    timestamp_ms: float
    sections: tuple[RichMetricSection, ...]
    evaluations: tuple[DashboardEvaluationRow, ...]


@dataclass(frozen=True, slots=True)
class QtMetricTableModel:
    """不依赖 Qt 运行时的指标表模型。"""

    columns: tuple[QtMetricColumn, ...]
    rows: tuple[DashboardMetricRow, ...]


@dataclass(frozen=True, slots=True)
class QtEvaluationTableModel:
    """不依赖 Qt 运行时的 canonical evaluation 表模型。"""

    columns: tuple[QtEvaluationColumn, ...]
    rows: tuple[DashboardEvaluationRow, ...]


@dataclass(frozen=True, slots=True)
class QtDashboardModel:
    """Qt 控件层可直接消费的不可变 dashboard 模型。"""

    schema_version: int
    run_id: str
    timestamp_ms: float
    metrics: QtMetricTableModel
    evaluations: QtEvaluationTableModel


_SECTION_TITLES: tuple[tuple[DashboardSection, str], ...] = (
    (DashboardSection.TRAINING, "Training"),
    (DashboardSection.PERCEPTION, "Perception"),
    (DashboardSection.TRACKING, "Tracking"),
    (DashboardSection.OUTCOME, "Outcome"),
    (DashboardSection.DECISION, "Decision"),
    (DashboardSection.RESOURCES, "Resources"),
)

_METRIC_SPECS: tuple[_MetricSpec, ...] = (
    _MetricSpec(
        DashboardMetric.STEP,
        DashboardSection.TRAINING,
        "Step",
        "",
        0,
        lambda snapshot: None if snapshot.metrics is None else snapshot.metrics.step,
    ),
    _MetricSpec(
        DashboardMetric.LOSS,
        DashboardSection.TRAINING,
        "Loss",
        "",
        6,
        lambda snapshot: None if snapshot.metrics is None else snapshot.metrics.loss,
    ),
    _MetricSpec(
        DashboardMetric.SCORE,
        DashboardSection.TRAINING,
        "Score",
        "",
        6,
        lambda snapshot: None if snapshot.metrics is None else snapshot.metrics.score,
    ),
    _MetricSpec(
        DashboardMetric.PERCEPTION_RECALL,
        DashboardSection.PERCEPTION,
        "Perception recall",
        "",
        6,
        lambda snapshot: (
            None if snapshot.metrics is None else snapshot.metrics.perception_recall
        ),
    ),
    _MetricSpec(
        DashboardMetric.TRACKING_ID_SWITCHES,
        DashboardSection.TRACKING,
        "Tracking ID switches",
        "",
        0,
        lambda snapshot: (
            None if snapshot.metrics is None else snapshot.metrics.tracking_id_switches
        ),
    ),
    _MetricSpec(
        DashboardMetric.OUTCOME_NLL,
        DashboardSection.OUTCOME,
        "Outcome NLL",
        "",
        6,
        lambda snapshot: (
            None if snapshot.metrics is None else snapshot.metrics.outcome_nll
        ),
    ),
    _MetricSpec(
        DashboardMetric.OUTCOME_BRIER,
        DashboardSection.OUTCOME,
        "Outcome Brier",
        "",
        6,
        lambda snapshot: (
            None if snapshot.metrics is None else snapshot.metrics.outcome_brier
        ),
    ),
    _MetricSpec(
        DashboardMetric.OUTCOME_ECE,
        DashboardSection.OUTCOME,
        "Outcome ECE",
        "",
        6,
        lambda snapshot: (
            None if snapshot.metrics is None else snapshot.metrics.outcome_ece
        ),
    ),
    _MetricSpec(
        DashboardMetric.EXPECTED_SCORE_ERROR,
        DashboardSection.OUTCOME,
        "Expected score error",
        "",
        6,
        lambda snapshot: (
            None if snapshot.metrics is None else snapshot.metrics.expected_score_error
        ),
    ),
    _MetricSpec(
        DashboardMetric.DECISION_UTILITY,
        DashboardSection.DECISION,
        "Decision utility",
        "",
        6,
        lambda snapshot: (
            None if snapshot.metrics is None else snapshot.metrics.decision_utility
        ),
    ),
    _MetricSpec(
        DashboardMetric.WAIT_CLICK_RATIO,
        DashboardSection.DECISION,
        "Wait/click ratio",
        "",
        6,
        lambda snapshot: (
            None if snapshot.metrics is None else snapshot.metrics.wait_click_ratio
        ),
    ),
    _MetricSpec(
        DashboardMetric.THROUGHPUT,
        DashboardSection.RESOURCES,
        "Throughput",
        "samples/s",
        3,
        lambda snapshot: (
            None if snapshot.resources is None else snapshot.resources.throughput
        ),
    ),
    _MetricSpec(
        DashboardMetric.GPU_UTILIZATION,
        DashboardSection.RESOURCES,
        "GPU utilization",
        "%",
        2,
        lambda snapshot: (
            None
            if snapshot.resources is None
            else snapshot.resources.gpu_utilization * 100.0
        ),
    ),
    _MetricSpec(
        DashboardMetric.VRAM_USED_MB,
        DashboardSection.RESOURCES,
        "VRAM used",
        "MiB",
        2,
        lambda snapshot: (
            None if snapshot.resources is None else snapshot.resources.vram_used_mb
        ),
    ),
    _MetricSpec(
        DashboardMetric.VRAM_TOTAL_MB,
        DashboardSection.RESOURCES,
        "VRAM total",
        "MiB",
        2,
        lambda snapshot: (
            None if snapshot.resources is None else snapshot.resources.vram_total_mb
        ),
    ),
)

_QT_METRIC_COLUMNS: tuple[QtMetricColumn, ...] = tuple(QtMetricColumn)
_QT_EVALUATION_COLUMNS: tuple[QtEvaluationColumn, ...] = tuple(QtEvaluationColumn)


def _format_metric(value: MetricNumber | None, precision: int) -> str:
    """仅格式化已经存在的 telemetry 值，不推导或填补业务指标。"""

    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}f}"


def _project_metric_rows(
    snapshot: DashboardSnapshot,
) -> tuple[DashboardMetricRow, ...]:
    """按唯一规格表产生稳定、有完整指标槽位的行。"""

    return tuple(
        DashboardMetricRow(
            rank=rank,
            metric=spec.metric,
            section=spec.section,
            label=spec.label,
            value=value,
            display_value=_format_metric(value, spec.precision),
            unit=spec.unit,
        )
        for rank, spec in enumerate(_METRIC_SPECS)
        for value in (spec.extract(snapshot),)
    )


def _project_evaluations(
    snapshot: DashboardSnapshot,
) -> tuple[DashboardEvaluationRow, ...]:
    """保留 reporter snapshot 内 canonical event 的对象身份。"""

    if snapshot.evaluation is None:
        return ()
    return (DashboardEvaluationRow(snapshot.evaluation),)


def _require_snapshot(snapshot: DashboardSnapshot) -> None:
    """拒绝 mutable mapping/legacy state 等旁路输入。"""

    if not isinstance(snapshot, DashboardSnapshot):
        raise TypeError("snapshot 必须是 DashboardSnapshot")


class RichDashboardRenderer:
    """把 snapshot 纯投影为终端分区模型；实例本身不保存状态。"""

    @staticmethod
    def render(snapshot: DashboardSnapshot) -> RichDashboardModel:
        """返回确定性的不可变 Rich view-model。"""

        _require_snapshot(snapshot)
        rows = _project_metric_rows(snapshot)
        sections = tuple(
            RichMetricSection(
                section=section,
                title=title,
                rows=tuple(row for row in rows if row.section is section),
            )
            for section, title in _SECTION_TITLES
        )
        return RichDashboardModel(
            schema_version=snapshot.schema_version,
            run_id=snapshot.run_id,
            timestamp_ms=snapshot.timestamp_ms,
            sections=sections,
            evaluations=_project_evaluations(snapshot),
        )


class QtDashboardRenderer:
    """把 snapshot 纯投影为 Qt 表模型；不导入或调用 Qt。"""

    @staticmethod
    def render(snapshot: DashboardSnapshot) -> QtDashboardModel:
        """返回确定性的不可变 Qt view-model。"""

        _require_snapshot(snapshot)
        return QtDashboardModel(
            schema_version=snapshot.schema_version,
            run_id=snapshot.run_id,
            timestamp_ms=snapshot.timestamp_ms,
            metrics=QtMetricTableModel(
                columns=_QT_METRIC_COLUMNS,
                rows=_project_metric_rows(snapshot),
            ),
            evaluations=QtEvaluationTableModel(
                columns=_QT_EVALUATION_COLUMNS,
                rows=_project_evaluations(snapshot),
            ),
        )


__all__ = (
    "DashboardEvaluationRow",
    "DashboardMetric",
    "DashboardMetricRow",
    "DashboardSection",
    "GalleryTargetOverlay",
    "GalleryFrameOverlay",
    "GalleryPredictionOverlay",
    "QtDashboardModel",
    "QtDashboardRenderer",
    "QtEvaluationColumn",
    "QtEvaluationTableModel",
    "QtMetricColumn",
    "QtMetricTableModel",
    "RichDashboardModel",
    "RichDashboardRenderer",
    "RichMetricSection",
    "build_gallery_frame_overlay",
    "project_gallery_target_overlays",
    "render_gallery_png",
)
