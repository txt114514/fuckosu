"""数据质量问题与唯一阻断语义。"""

from dataclasses import dataclass
from enum import Enum

from .common import JSONValue, require_identifier


class DataQualitySeverity(str, Enum):
    """质量问题展示严重度；是否阻断由独立字段决定。"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DataQualityIssue:
    """可定位、可解释的数据质量问题。"""

    code: str
    severity: DataQualitySeverity
    blocks_training: bool
    sample_id: str | None
    message: str
    details: tuple[tuple[str, JSONValue], ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.code, "code")
        if self.sample_id is not None:
            require_identifier(self.sample_id, "sample_id")
        if not self.message or self.message != self.message.strip():
            raise ValueError("message 不得为空且不得有首尾空格")
        keys = tuple(key for key, _ in self.details)
        if any(not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("details 的键必须非空且唯一")


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """质量门报告；ok 只由 blocks_training 推导。"""

    issues: tuple[DataQualityIssue, ...]

    @property
    def ok(self) -> bool:
        """仅按 ``blocks_training`` 汇总唯一的质量门结论。"""

        return not any(issue.blocks_training for issue in self.issues)
