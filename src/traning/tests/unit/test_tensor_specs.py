"""TensorSpec 集中验证 shape、dtype、device 和有限性。"""

from __future__ import annotations

import pytest
import torch

from traning.lib.validation import (
    TensorSpec,
    require_same_device,
    require_same_dtype,
    require_shape,
    require_tensor,
)


def test_tensor_spec_returns_same_tensor_after_complete_boundary_validation() -> None:
    """校验不复制、不搬运 Tensor，内部可继续使用原对象。"""

    tensor = torch.ones((2, 3, 4, 5), dtype=torch.float32)
    spec = TensorSpec(
        shape=(None, 3, 4, 5),
        dtype=torch.float32,
        device="cpu",
        finite=True,
    )
    assert require_tensor(tensor, "frames", spec=spec) is tensor
    assert require_shape(tensor, (2, None, 4, 5), "frames") == (2, 3, 4, 5)


def test_tensor_spec_rejects_wrong_shape_dtype_empty_and_non_finite() -> None:
    """所有昂贵检查由一份规格表在边界完成。"""

    tensor = torch.ones((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        TensorSpec(shape=(2, 4)).validate(tensor, "features")
    with pytest.raises(TypeError, match="dtype"):
        TensorSpec(dtype=torch.float64).validate(tensor, "features")
    with pytest.raises(ValueError, match="empty"):
        TensorSpec(shape=(0, 3)).validate(torch.empty((0, 3)), "features")
    with pytest.raises(ValueError, match="finite"):
        TensorSpec().validate(torch.tensor((1.0, float("inf"))), "features")
    with pytest.raises(TypeError, match="torch.Tensor"):
        require_tensor([1.0, 2.0], "features")


def test_same_device_and_dtype_helpers_return_the_shared_property() -> None:
    """跨张量约束集中为可复用 helper。"""

    first = torch.ones((1,), dtype=torch.float32)
    second = torch.zeros((2,), dtype=torch.float32)
    assert require_same_device(first, second) == torch.device("cpu")
    assert require_same_dtype(first, second) is torch.float32
    with pytest.raises(TypeError, match="one dtype"):
        require_same_dtype(first, second.double())
    with pytest.raises(ValueError, match="at least one"):
        require_same_device()
