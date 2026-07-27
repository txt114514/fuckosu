"""重导出带提交语义的图集输出编号分配 API。"""

from traning.lib.visualization.output_identity import (
    allocate_output_identity,
    reserve_output_identity_for_commit,
)

__all__ = ["allocate_output_identity", "reserve_output_identity_for_commit"]
