from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DynamicSparseLinear(nn.Module):
    """Linear layer with a deterministic top-k dynamic sparse weight mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        sparsity: float = 0.50,
        update_interval: int = 16,
        min_density: float = 0.05,
    ) -> None:
        super().__init__()
        if min(in_features, out_features, update_interval) <= 0:
            raise ValueError("sparse linear dimensions and interval must be positive")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1)")
        if not 0.0 < min_density <= 1.0:
            raise ValueError("min_density must be in (0, 1]")
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.update_interval = update_interval
        self.min_density = min_density
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("mask", torch.ones(out_features, in_features))
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))
        self.reset_parameters()
        self.refresh_mask()

    @property
    def density(self) -> float:
        return max(self.min_density, 1.0 - self.sparsity)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def refresh_mask(self) -> None:
        self.mask.copy_(self._mask_from_weight().to(self.mask.dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mask = self._mask_from_weight() if self.training else self.mask
        return F.linear(input, self.weight * mask, self.bias)

    def _mask_from_weight(self) -> torch.Tensor:
        keep = max(1, int(round(self.weight.numel() * self.density)))
        weight_abs = self.weight.detach().abs()
        flat = weight_abs.flatten()
        if keep >= flat.numel():
            return torch.ones_like(self.weight)
        threshold = torch.topk(flat, keep, sorted=False).values.min()
        return (weight_abs >= threshold).to(self.weight.dtype)


def maybe_sparse_linear(
    in_features: int,
    out_features: int,
    *,
    enabled: bool,
    bias: bool = True,
    sparsity: float = 0.50,
    update_interval: int = 16,
    min_density: float = 0.05,
) -> nn.Module:
    if not enabled:
        return nn.Linear(in_features, out_features, bias=bias)
    return DynamicSparseLinear(
        in_features,
        out_features,
        bias=bias,
        sparsity=sparsity,
        update_interval=update_interval,
        min_density=min_density,
    )


__all__ = ["DynamicSparseLinear", "maybe_sparse_linear"]
