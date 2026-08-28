"""V2 训练、评分与 gallery 共用的原帧坐标适配器。

这里只负责绑定“已标定仿射变换 + 标定时原帧尺寸”；真正的
osu! 与视频坐标方程始终由 :mod:`package` 的公开 API 执行。这样训练
target、canonical scoring 和 gallery 不会各自拥有一套系数或补偿偏移。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from package import (
    AffineOsuVideoTransform,
    CoordinateTransformSpec,
    ImageSize,
    OSU_PLAYFIELD_HEIGHT,
    OSU_PLAYFIELD_WIDTH,
    OsuVideoCoordinateTransform,
    coordinate_transform_fingerprint,
)
from traning.contracts import Point2D, require_transform_fingerprint


def _require_finite(value: float, field_name: str) -> None:
    """校验坐标是有限实数，并显式拒绝布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} 必须是数值")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} 必须是有限数值")


def _require_frame_size(width: int, height: int) -> None:
    """校验原帧尺寸，避免 ``bool`` 被当成整数接受。"""

    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
    ):
        raise TypeError("source frame width/height 必须是整数")
    if width < 1 or height < 1:
        raise ValueError("source frame width/height 必须为正整数")


@dataclass(frozen=True, slots=True)
class OsuPoint:
    """osu!standard playfield 内的有界坐标。"""

    x: float
    y: float

    def __post_init__(self) -> None:
        """拒绝非有限值和 playfield 边界外的点。"""

        _require_finite(self.x, "osu.x")
        _require_finite(self.y, "osu.y")
        if not 0.0 <= self.x <= OSU_PLAYFIELD_WIDTH:
            raise ValueError("osu.x 超出 playfield 边界")
        if not 0.0 <= self.y <= OSU_PLAYFIELD_HEIGHT:
            raise ValueError("osu.y 超出 playfield 边界")


@dataclass(frozen=True, slots=True)
class CanonicalScoringPoint:
    """由原帧逆变换得到、允许落在 playfield 外的评分坐标。

    原帧边缘点击可能合法地映射到负 osu! 坐标。它们应由 scorer 计为空间
    miss，而不是让整段评估异常退出；变换指纹仍随点保留以防混用。
    """

    x: float
    y: float
    transform_fingerprint: str

    def __post_init__(self) -> None:
        _require_finite(self.x, "canonical_scoring.x")
        _require_finite(self.y, "canonical_scoring.y")
        require_transform_fingerprint(self.transform_fingerprint)


@dataclass(frozen=True, slots=True)
class FramePixelPoint:
    """与具体原帧尺寸和坐标变换指纹绑定的有界像素坐标。"""

    x: float
    y: float
    source_frame_width: int
    source_frame_height: int
    transform_fingerprint: str

    def __post_init__(self) -> None:
        """将尺寸与边界校验放在领域对象边界。"""

        _require_frame_size(self.source_frame_width, self.source_frame_height)
        require_transform_fingerprint(self.transform_fingerprint)
        _require_finite(self.x, "frame.x")
        _require_finite(self.y, "frame.y")
        if not 0.0 <= self.x < self.source_frame_width:
            raise ValueError("frame.x 超出原帧像素边界")
        if not 0.0 <= self.y < self.source_frame_height:
            raise ValueError("frame.y 超出原帧像素边界")


@dataclass(frozen=True, slots=True)
class FrameCoordinateTransform:
    """将一个共享仿射变换与标定原帧的尺寸、身份和指纹绑定。

    该适配器不会根据当前图像猜测居中 playfield，也不会在尺寸错误
    时缩放或偏移坐标。任何 schema/尺寸不一致都是硬错误。
    """

    source_frame_width: int
    source_frame_height: int
    transform_identity: str
    transform: OsuVideoCoordinateTransform = field(repr=False, compare=False)
    transform_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        """验证变换来源，并用共享规格 API 生成稳定指纹。"""

        _require_frame_size(self.source_frame_width, self.source_frame_height)
        if (
            not isinstance(self.transform_identity, str)
            or not self.transform_identity
            or self.transform_identity != self.transform_identity.strip()
        ):
            raise ValueError("transform_identity 必须是非空且无首尾空格的字符串")
        if not isinstance(self.transform, AffineOsuVideoTransform):
            raise TypeError("必须显式提供已标定的 AffineOsuVideoTransform")

        # 指纹同时包含仿射矩阵与标定原帧尺寸，避免误用旧缓存。
        affine_spec = self.transform.spec(
            source=self.transform_identity,
            status="calibrated",
        )
        bound_spec = CoordinateTransformSpec(
            version=affine_spec.version,
            rect=affine_spec.rect,
            source=affine_spec.source,
            transform_status=affine_spec.transform_status,
            source_size=ImageSize(
                width=float(self.source_frame_width),
                height=float(self.source_frame_height),
            ),
            matrix=affine_spec.matrix,
        )
        object.__setattr__(
            self,
            "transform_fingerprint",
            coordinate_transform_fingerprint(bound_spec),
        )

        # 连同四个 playfield 角点验证标定结果，不允许构造后再静默 clamp。
        for osu_point in (
            OsuPoint(0.0, 0.0),
            OsuPoint(OSU_PLAYFIELD_WIDTH, 0.0),
            OsuPoint(0.0, OSU_PLAYFIELD_HEIGHT),
            OsuPoint(OSU_PLAYFIELD_WIDTH, OSU_PLAYFIELD_HEIGHT),
        ):
            self._transform_osu_to_frame(osu_point)

    def _require_bound_size(self, width: int, height: int) -> None:
        """要求消费者声明的原帧尺寸与标定尺寸完全一致。"""

        _require_frame_size(width, height)
        if width != self.source_frame_width or height != self.source_frame_height:
            raise ValueError("source frame size 与坐标变换标定尺寸不一致")

    def _transform_osu_to_frame(self, point: OsuPoint) -> FramePixelPoint:
        """统一的 osu -> 原帧实现；三个消费者不得自行算系数。"""

        x, y = self.transform.osu_to_video(point.x, point.y)
        return FramePixelPoint(
            x=x,
            y=y,
            source_frame_width=self.source_frame_width,
            source_frame_height=self.source_frame_height,
            transform_fingerprint=self.transform_fingerprint,
        )

    def bind_frame_prediction(
        self,
        *,
        x: float,
        y: float,
        source_frame_width: int,
        source_frame_height: int,
    ) -> FramePixelPoint:
        """把 runtime 原帧预测绑定到本变换指纹，供 canonical scoring 使用。"""

        self._require_bound_size(source_frame_width, source_frame_height)
        return FramePixelPoint(
            x=x,
            y=y,
            source_frame_width=source_frame_width,
            source_frame_height=source_frame_height,
            transform_fingerprint=self.transform_fingerprint,
        )

    def ground_truth_to_training_target(
        self,
        point: OsuPoint,
        *,
        source_frame_width: int,
        source_frame_height: int,
    ) -> FramePixelPoint:
        """把 osu GT 转成稠密感知训练使用的原帧像素 target。"""

        self._require_bound_size(source_frame_width, source_frame_height)
        return self._transform_osu_to_frame(point)

    def ground_truth_radius_to_training_target(
        self,
        radius_osu: float,
        *,
        source_frame_width: int,
        source_frame_height: int,
    ) -> float:
        """把 osu! 半径投影成与同一标定绑定的原帧像素半径。"""

        self._require_bound_size(source_frame_width, source_frame_height)
        _require_finite(radius_osu, "radius_osu")
        if radius_osu <= 0.0:
            raise ValueError("radius_osu 必须大于 0")
        radius_px = self.transform.osu_radius_to_video(float(radius_osu))
        _require_finite(radius_px, "radius_px")
        if radius_px <= 0.0:
            raise ValueError("标定后的 radius_px 必须大于 0")
        return radius_px

    def ground_truth_direction_to_training_target(
        self,
        start: Point2D,
        end: Point2D,
        *,
        source_frame_width: int,
        source_frame_height: int,
    ) -> tuple[float, float]:
        """把 osu! 路径首段转换为原帧单位方向，允许控制点越过边界。

        osu! slider 的 Bézier/Catmull 控制点可以合法位于 512×384 playfield
        外部。方向只依赖 affine 的线性部分，不应把控制点构造成有界
        ``OsuPoint``，也不应把越界点裁剪后改变监督方向。
        """

        self._require_bound_size(source_frame_width, source_frame_height)
        if not isinstance(start, Point2D) or not isinstance(end, Point2D):
            raise TypeError("slider start/end 必须是 Point2D")
        start_x, start_y = self.transform.osu_to_video(start.x, start.y)
        end_x, end_y = self.transform.osu_to_video(end.x, end.y)
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        norm = math.hypot(delta_x, delta_y)
        if norm <= 0.0:
            raise ValueError("slider path 首段不得是零长度")
        return delta_x / norm, delta_y / norm

    def prediction_to_canonical_scoring(
        self,
        point: FramePixelPoint,
    ) -> CanonicalScoringPoint:
        """把原帧预测逆变换为可落在 playfield 外的评分坐标。"""

        self._require_bound_size(
            point.source_frame_width,
            point.source_frame_height,
        )
        if point.transform_fingerprint != self.transform_fingerprint:
            raise ValueError(
                "frame prediction 与 canonical scoring 的坐标变换指纹不一致"
            )
        x, y = self.transform.video_to_osu(point.x, point.y)
        return CanonicalScoringPoint(
            x=x,
            y=y,
            transform_fingerprint=self.transform_fingerprint,
        )

    def target_to_gallery_overlay(
        self,
        point: OsuPoint,
        *,
        source_frame_width: int,
        source_frame_height: int,
    ) -> FramePixelPoint:
        """把 osu target 转成 gallery 在原帧上绘制的像素点。"""

        self._require_bound_size(source_frame_width, source_frame_height)
        return self._transform_osu_to_frame(point)


__all__ = (
    "CanonicalScoringPoint",
    "FrameCoordinateTransform",
    "FramePixelPoint",
    "OsuPoint",
)
