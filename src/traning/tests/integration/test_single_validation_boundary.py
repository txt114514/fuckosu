"""边界完整校验一次，内部默认只消费 Trusted 值。"""

from __future__ import annotations

import torch

from traning.lib.validation import (
    ModelInputBoundary,
    TensorSpec,
    unwrap_trusted,
    validate_internal,
)


def test_same_boundary_does_not_repeat_full_tensor_validation() -> None:
    """同一个可信对象二次经过模型入口时不会再次扫描 Tensor。"""

    calls = 0
    spec = TensorSpec(shape=(None, 3, 8, 8), dtype=torch.float32)

    def _validate_frames(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return spec.validate(value, "frames")

    boundary = ModelInputBoundary(_validate_frames)
    frames = torch.ones((2, 3, 8, 8))
    verified = boundary.validate(frames)
    repeated = boundary.validate(verified)

    assert repeated is verified
    assert unwrap_trusted(repeated, boundary="model_input") is frames
    assert calls == 1


def test_internal_full_check_is_opt_in(monkeypatch) -> None:
    """默认热路径跳过内部完整检查，诊断环境变量可显式恢复。"""

    calls = 0

    def _validator(value: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        TensorSpec(shape=(1,), dtype=torch.float32).validate(value)

    tensor = torch.ones((1,))
    monkeypatch.delenv("TRANING_STRICT_INTERNAL_CHECKS", raising=False)
    assert validate_internal(tensor, _validator) is tensor
    assert calls == 0

    monkeypatch.setenv("TRANING_STRICT_INTERNAL_CHECKS", "1")
    assert validate_internal(tensor, _validator) is tensor
    assert calls == 1
