"""定义候选缓存当前协议及仅供诊断/迁移读取的历史协议。"""

CANDIDATE_CACHE_VERSION = "spatial-candidate-cache-v2"
SUPPORTED_CANDIDATE_CACHE_VERSIONS = frozenset(
    {"spatial-candidate-cache-v1", CANDIDATE_CACHE_VERSION}
)

__all__ = [
    "CANDIDATE_CACHE_VERSION",
    "SUPPORTED_CANDIDATE_CACHE_VERSIONS",
]
