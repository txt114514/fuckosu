"""已弃用兼容转发；新代码必须导入对应的 conf、core、lib 或 state 路径。"""

from traning.state.common import (
    JSONObject,
    JSONScalar,
    JSONValue,
    require_finite,
    require_identifier,
    require_nonnegative,
    require_probability,
    require_probability_sum,
    require_sha256,
    require_transform_fingerprint,
)

__deprecated__ = True
__all__ = [
    "JSONObject",
    "JSONScalar",
    "JSONValue",
    "require_finite",
    "require_identifier",
    "require_nonnegative",
    "require_probability",
    "require_probability_sum",
    "require_sha256",
    "require_transform_fingerprint",
]
