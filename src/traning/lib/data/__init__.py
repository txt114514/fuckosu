"""训练数据的标注解析与完整帧读取公共入口。

固定 Patch/tiling API 仍可从其 deprecated leaf module 显式导入，但不再属于
默认公开面，防止新训练或推理代码无意间重新接入旧分块主流程。
"""

from traning.lib.data.annotation import (
    HitObjectAnnotation,
    SegmentAnnotation,
    load_annotation,
    visible_hit_objects,
)
from traning.lib.data.color_cues import (
    ColorCueMode,
    append_color_cues,
    color_cue_channel_count,
    extract_osu_basic_color_cues,
)
from traning.lib.data.discovery import discover_segments
from traning.lib.data.models import (
    DatasetIssue,
    DiscoveryResult,
    FrameReference,
    SegmentRecord,
)
from traning.lib.data.synthetic_structures import (
    SyntheticStructure,
    make_boundary_circle,
    make_cross_patch_ring,
    make_cross_patch_slider,
    make_noise_background,
    make_spinner,
)
from traning.lib.data.video_reader import VideoReader

__all__ = [
    "ColorCueMode",
    "DatasetIssue",
    "DiscoveryResult",
    "FrameReference",
    "HitObjectAnnotation",
    "SegmentAnnotation",
    "SegmentRecord",
    "SyntheticStructure",
    "VideoReader",
    "append_color_cues",
    "color_cue_channel_count",
    "discover_segments",
    "extract_osu_basic_color_cues",
    "load_annotation",
    "make_boundary_circle",
    "make_cross_patch_ring",
    "make_cross_patch_slider",
    "make_noise_background",
    "make_spinner",
    "visible_hit_objects",
]
