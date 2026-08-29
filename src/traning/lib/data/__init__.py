"""训练数据的标注解析、帧读取、分块和坐标工具公共入口。"""

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
from traning.lib.data.coordinates import (
    feature_grid_to_image,
    global_to_local,
    global_to_patch_indices,
    image_to_feature_grid,
    local_to_global,
)
from traning.lib.data.discovery import discover_segments
from traning.lib.data.models import (
    DatasetIssue,
    DiscoveryResult,
    FrameReference,
    SegmentRecord,
)
from traning.lib.data.tiling import (
    PatchWindow,
    build_patch_windows,
    iter_patches,
)
from traning.lib.data.patch_stream import PatchMeta, PatchStream
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
    "PatchMeta",
    "PatchStream",
    "PatchWindow",
    "SegmentAnnotation",
    "SegmentRecord",
    "SyntheticStructure",
    "VideoReader",
    "append_color_cues",
    "build_patch_windows",
    "color_cue_channel_count",
    "discover_segments",
    "extract_osu_basic_color_cues",
    "feature_grid_to_image",
    "global_to_local",
    "global_to_patch_indices",
    "image_to_feature_grid",
    "iter_patches",
    "load_annotation",
    "local_to_global",
    "make_boundary_circle",
    "make_cross_patch_ring",
    "make_cross_patch_slider",
    "make_noise_background",
    "make_spinner",
    "visible_hit_objects",
]
