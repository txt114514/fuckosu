from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from traning.conf import Settings
from traning.core.temporal.dataset import (
    IGNORE_CANDIDATE_ID,
    NO_OP_ACTION_ID,
    TemporalCandidateWindowDataset,
    TemporalWindow,
)
from traning.lib.models import CausalTemporalModel
from traning.lib.runtime import (
    CudaRuntimeConfig,
    autocast_context,
    collect_memory_snapshot,
    configure_torch_runtime,
    create_grad_scaler,
    enforce_runtime_memory_budget,
    maybe_compile_module,
    module_to_device,
    tensor_to_device,
)


@dataclass(frozen=True)
class TemporalTrainingResult:
    run_dir: Path
    checkpoint_path: Path
    device: str
    steps: int
    windows: int
    sequence_length: int
    candidate_slots: int
    input_size: int
    final_loss: float
    action_loss: float
    candidate_loss: float
    xy_loss: float
    time_loss: float
    target_strategy: str
    cuda_max_allocated_gib: float | None
    cuda_max_reserved_gib: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "steps": self.steps,
            "windows": self.windows,
            "sequence_length": self.sequence_length,
            "candidate_slots": self.candidate_slots,
            "input_size": self.input_size,
            "final_loss": self.final_loss,
            "action_loss": self.action_loss,
            "candidate_loss": self.candidate_loss,
            "xy_loss": self.xy_loss,
            "time_loss": self.time_loss,
            "target_strategy": self.target_strategy,
            "cuda_max_allocated_gib": self.cuda_max_allocated_gib,
            "cuda_max_reserved_gib": self.cuda_max_reserved_gib,
        }


def run_temporal_training(
    settings: Settings,
    *,
    cache_dir: Path,
    device: torch.device,
    run_dir: Path,
    max_steps: int = 1,
    learning_rate: float = 1e-4,
    sequence_length: int | None = None,
    candidate_slots: int | None = None,
    dataset: Sequence[TemporalWindow] | None = None,
) -> TemporalTrainingResult:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    selected_sequence_length = sequence_length or settings.temporal.history_frames
    selected_candidate_slots = (
        candidate_slots or settings.candidate_cache.max_candidates_per_frame
    )
    source = (
        dataset
        if dataset is not None
        else TemporalCandidateWindowDataset.from_cache_dir(
            cache_dir,
            sequence_length=selected_sequence_length,
            candidate_slots=selected_candidate_slots,
        )
    )
    if len(source) <= 0:
        raise ValueError("temporal training dataset must not be empty")
    first = source[0]
    input_size = int(first.features.shape[-1])
    model = CausalTemporalModel(
        input_size=input_size,
        hidden_size=settings.temporal.hidden_size,
        layers=settings.temporal.layers,
        candidate_slots=selected_candidate_slots,
        action_classes=4,
    )
    enforce_runtime_memory_budget(
        device=device,
        max_vram_gib=settings.memory.max_vram_gib,
        reserve_vram_gib=settings.memory.reserve_vram_gib,
        max_ram_gib=settings.memory.max_ram_gib,
        reserve_ram_gib=settings.memory.reserve_ram_gib,
    )
    runtime_state = configure_torch_runtime(
        device=device,
        amp_dtype=settings.memory.amp_dtype,
        runtime=CudaRuntimeConfig(
            allow_tf32=settings.memory.allow_tf32,
            cudnn_benchmark=settings.memory.cudnn_benchmark,
            matmul_float32_precision=settings.memory.matmul_float32_precision,
            channels_last=False,
        ),
    )
    model = module_to_device(model, device, channels_last=False)
    model = maybe_compile_module(model, enabled=settings.memory.compile_model)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = create_grad_scaler(
        device=device,
        amp_dtype=settings.memory.amp_dtype,
        mode=settings.memory.grad_scaler,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    metrics = {
        "loss": 0.0,
        "action_loss": 0.0,
        "candidate_loss": 0.0,
        "xy_loss": 0.0,
        "time_loss": 0.0,
    }
    for step in range(max_steps):
        window = source[step % len(source)]
        batch = _window_to_device(window, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, runtime_state.amp_dtype):
            outputs, _ = model(batch["features"])
            loss, loss_parts = _compute_temporal_loss(
                outputs,
                action_target=batch["action_target"],
                selected_candidate_target=batch["selected_candidate_target"],
                xy_target=batch["xy_target"],
                time_offset_target=batch["time_offset_target"],
                frame_mask=batch["frame_mask"],
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        metrics = {
            "loss": float(loss.detach().cpu()),
            **{key: float(value.detach().cpu()) for key, value in loss_parts.items()},
        }
    snapshot = collect_memory_snapshot()
    result = TemporalTrainingResult(
        run_dir=run_dir,
        checkpoint_path=run_dir / "temporal_model.pt",
        device=str(device),
        steps=max_steps,
        windows=len(source),
        sequence_length=selected_sequence_length,
        candidate_slots=selected_candidate_slots,
        input_size=input_size,
        final_loss=metrics["loss"],
        action_loss=metrics["action_loss"],
        candidate_loss=metrics["candidate_loss"],
        xy_loss=metrics["xy_loss"],
        time_loss=metrics["time_loss"],
        target_strategy=first.target_strategy,
        cuda_max_allocated_gib=snapshot.max_allocated_gib,
        cuda_max_reserved_gib=snapshot.max_reserved_gib,
    )
    _write_summary(result)
    _write_checkpoint(
        result,
        model=model,
        hidden_size=settings.temporal.hidden_size,
        layers=settings.temporal.layers,
    )
    return result


def _window_to_device(
    window: TemporalWindow,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "features": tensor_to_device(
            window.features.unsqueeze(1),
            device,
            channels_last=False,
            non_blocking=False,
        ),
        "action_target": window.action_target.to(device=device),
        "selected_candidate_target": window.selected_candidate_target.to(
            device=device
        ),
        "xy_target": window.xy_target.to(device=device),
        "time_offset_target": window.time_offset_target.to(device=device),
        "frame_mask": window.frame_mask.to(device=device),
    }


def _compute_temporal_loss(
    outputs,
    *,
    action_target: torch.Tensor,
    selected_candidate_target: torch.Tensor,
    xy_target: torch.Tensor,
    time_offset_target: torch.Tensor,
    frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    action_logits = torch.cat([output.action_logits for output in outputs], dim=0)
    candidate_logits = torch.cat(
        [output.selected_candidate_logits for output in outputs],
        dim=0,
    )
    xy_prediction = torch.cat([output.x for output in outputs], dim=0)
    y_prediction = torch.cat([output.y for output in outputs], dim=0)
    time_prediction = torch.cat([output.time_offset_ms for output in outputs], dim=0)
    valid = frame_mask.bool()
    if not bool(valid.any()):
        raise ValueError("temporal loss requires at least one valid frame")
    action_loss = F.cross_entropy(action_logits[valid], action_target[valid])
    candidate_frames = valid & (selected_candidate_target != IGNORE_CANDIDATE_ID)
    if bool(candidate_frames.any()):
        candidate_loss = F.cross_entropy(
            candidate_logits[candidate_frames],
            selected_candidate_target[candidate_frames],
        )
    else:
        candidate_loss = action_logits.sum() * 0.0
    action_frames = valid & (action_target != NO_OP_ACTION_ID)
    if bool(action_frames.any()):
        xy = torch.cat([xy_prediction, y_prediction], dim=1)
        xy_loss = F.mse_loss(xy[action_frames], xy_target[action_frames])
        time_loss = F.mse_loss(
            time_prediction[action_frames],
            time_offset_target[action_frames],
        )
    else:
        xy_loss = action_logits.sum() * 0.0
        time_loss = action_logits.sum() * 0.0
    loss = action_loss + candidate_loss + xy_loss + 0.01 * time_loss
    return loss, {
        "action_loss": action_loss,
        "candidate_loss": candidate_loss,
        "xy_loss": xy_loss,
        "time_loss": time_loss,
    }


def _write_summary(result: TemporalTrainingResult) -> None:
    result.run_dir.mkdir(parents=True, exist_ok=True)
    (result.run_dir / "summary.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_checkpoint(
    result: TemporalTrainingResult,
    *,
    model: torch.nn.Module,
    hidden_size: int,
    layers: int,
) -> None:
    result.run_dir.mkdir(parents=True, exist_ok=True)
    state_source = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "version": "temporal-decision-v1",
            "model_state": state_source.state_dict(),
            "model_config": {
                "input_size": result.input_size,
                "hidden_size": hidden_size,
                "layers": layers,
                "candidate_slots": result.candidate_slots,
                "action_classes": 4,
            },
            "sequence_length": result.sequence_length,
            "target_strategy": result.target_strategy,
        },
        result.checkpoint_path,
    )


__all__ = [
    "TemporalTrainingResult",
    "run_temporal_training",
]
