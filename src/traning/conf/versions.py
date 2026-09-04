"""V2 配置与对应持久化实现共用的 schema 版本。"""

CANDIDATE_CACHE_SCHEMA_VERSION = 2
"""候选缓存必须包含坐标变换指纹的唯一受支持版本。"""

CALIBRATION_EVIDENCE_SCHEMA_VERSION = 1
"""坐标控制点证据与原始拟合集可用性声明的唯一受支持版本。"""

TELEMETRY_SCHEMA_VERSION = 2
"""evaluation 事件保留坐标来源与点击误差的唯一受支持版本。"""


__all__ = (
    "CALIBRATION_EVIDENCE_SCHEMA_VERSION",
    "CANDIDATE_CACHE_SCHEMA_VERSION",
    "TELEMETRY_SCHEMA_VERSION",
)
