from __future__ import annotations

import torch
from torch import nn

from traning.models.outputs import ActionPrediction


class CausalTemporalModel(nn.Module):
    """Causal GRU action head for streaming frame-by-frame inference."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 256,
        layers: int = 2,
        candidate_slots: int = 64,
        action_classes: int = 4,
    ) -> None:
        super().__init__()
        if min(input_size, hidden_size, layers, candidate_slots, action_classes) <= 0:
            raise ValueError("temporal model dimensions must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.cells = nn.ModuleList(
            nn.GRUCell(input_size if layer == 0 else hidden_size, hidden_size)
            for layer in range(layers)
        )
        self.action_head = nn.Linear(hidden_size, action_classes)
        self.candidate_head = nn.Linear(hidden_size, candidate_slots)
        self.xy_head = nn.Linear(hidden_size, 2)
        self.time_head = nn.Linear(hidden_size, 1)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        state_dtype = dtype or parameter.dtype
        return torch.zeros(
            self.layers,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=state_dtype,
        )

    def step(
        self,
        current_features: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> tuple[ActionPrediction, torch.Tensor]:
        if current_features.ndim != 2:
            raise ValueError("current_features must use BF layout")
        if previous_state.shape[:2] != (self.layers, current_features.shape[0]):
            raise ValueError("previous_state shape must be layers x batch x hidden")
        if previous_state.shape[2] != self.hidden_size:
            raise ValueError("previous_state hidden size mismatch")
        layer_input = current_features
        next_states = []
        for layer, cell in enumerate(self.cells):
            hidden = cell(layer_input, previous_state[layer])
            next_states.append(hidden)
            layer_input = hidden
        next_state = torch.stack(next_states, dim=0)
        xy = self.xy_head(layer_input)
        prediction = ActionPrediction(
            action_logits=self.action_head(layer_input),
            selected_candidate_logits=self.candidate_head(layer_input),
            x=xy[:, :1],
            y=xy[:, 1:2],
            time_offset_ms=self.time_head(layer_input),
            next_hidden_state=next_state,
        )
        return prediction, next_state

    def forward(
        self, sequence: torch.Tensor
    ) -> tuple[list[ActionPrediction], torch.Tensor]:
        if sequence.ndim != 3:
            raise ValueError("sequence must use TBF layout")
        _, batch_size, _ = sequence.shape
        state = self.initial_state(batch_size, sequence.device, dtype=sequence.dtype)
        outputs: list[ActionPrediction] = []
        for frame_features in sequence:
            output, state = self.step(frame_features, state)
            outputs.append(output)
        return outputs, state


__all__ = ["CausalTemporalModel"]
