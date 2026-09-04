"""训练和推理各入口的一次性边界校验对象。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from .trusted import Trusted


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class ValidationBoundary(Generic[_T]):
    """调用一个严格 validator 并产生带边界身份的 ``Trusted`` 值。

    validator 可以原地验证后返回 ``None``，也可以返回规范化后的同类型对象。同一个
    ``Trusted`` 对象再次经过同名边界时不会重复调用 validator。
    """

    validator: Callable[[_T], _T | None]

    boundary_name: ClassVar[str] = "validation"

    def validate(self, value: _T | Trusted[_T]) -> Trusted[_T]:
        """严格验证未标记值；同边界已验证值直接透传。"""

        if isinstance(value, Trusted):
            if value.boundary == self.boundary_name:
                return value
            candidate = value.value
        else:
            candidate = value
        normalized = self.validator(candidate)
        return Trusted(
            value=candidate if normalized is None else normalized,
            boundary=self.boundary_name,
        )

    def __call__(self, value: _T | Trusted[_T]) -> Trusted[_T]:
        """把边界对象作为 validator 调用。"""

        return self.validate(value)


class ConfigBoundary(ValidationBoundary[_T]):
    """配置文件或配置映射进入 typed config 的严格边界。"""

    boundary_name = "config"


class DataBoundary(ValidationBoundary[_T]):
    """磁盘、数据库或解码数据进入 typed dataset 的严格边界。"""

    boundary_name = "data"


class ModelInputBoundary(ValidationBoundary[_T]):
    """外部 batch 进入模型 forward 前的严格边界。"""

    boundary_name = "model_input"


class PerceptionBoundary(ValidationBoundary[_T]):
    """感知输入或输出进入正式因果链路的严格边界。"""

    boundary_name = "perception"


class TrackingBoundary(ValidationBoundary[_T]):
    """候选观测进入跟踪层的严格边界。"""

    boundary_name = "tracking"


class BeliefBoundary(ValidationBoundary[_T]):
    """跟踪状态进入因果 belief 层的严格边界。"""

    boundary_name = "belief"


class OutcomeBoundary(ValidationBoundary[_T]):
    """belief 进入 outcome 预测层的严格边界。"""

    boundary_name = "outcome"


class DecisionBoundary(ValidationBoundary[_T]):
    """outcome 分布进入动作规划层的严格边界。"""

    boundary_name = "decision"


class TrainingBatchBoundary(ValidationBoundary[_T]):
    """DataLoader batch 进入训练 step 的严格边界。"""

    boundary_name = "training_batch"


class LossBoundary(ValidationBoundary[_T]):
    """模型输出与标签进入 loss 计算前的严格边界。"""

    boundary_name = "loss"


__all__ = [
    "BeliefBoundary",
    "ConfigBoundary",
    "DataBoundary",
    "DecisionBoundary",
    "LossBoundary",
    "ModelInputBoundary",
    "OutcomeBoundary",
    "PerceptionBoundary",
    "TrackingBoundary",
    "TrainingBatchBoundary",
    "ValidationBoundary",
]
