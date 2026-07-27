"""重导出训练侧共享标注渲染器，确保坐标解析与回退策略只有一份实现。"""

from traning.lib.visualization.render import (
    render_annotated_frame,
    save_annotated_frame,
)

__all__ = ["render_annotated_frame", "save_annotated_frame"]
