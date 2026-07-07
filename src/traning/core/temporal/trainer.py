from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
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
from traning.lib.models import CausalTemporalModel, DynamicSparseLinear
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
from traning.lib.reporting import should_report_training_step
from traning.core.training_inheritance import (
    TrainingPosition,
    atomic_torch_save_checkpoint,
    build_training_checkpoint,
    load_training_checkpoint,
    restore_module_state,
    restore_rng_state,
)
from visualization.lib import (
    DatasetUsageState,
    NullReporter,
    PipelineStageState,
    ResourceState,
    TrainingEvent,
    TrainingReporter,
    collect_resource_state,
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
    loss_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "action": 1.0,
            "candidate": 1.0,
            "xy": 1.0,
            "time_offset": 0.01,
        }
    )

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
            "loss_weights": dict(self.loss_weights),
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
    reporter: TrainingReporter = NullReporter(),
    resume_checkpoint_path: Path | None = None,
    resume_policy: str = "none",
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
    torch.manual_seed(settings.runtime.seed)
    first = source[0]
    input_size = int(first.features.shape[-1])
    model = CausalTemporalModel(
        input_size=input_size,
        hidden_size=settings.temporal.hidden_size,
        layers=settings.temporal.layers,
        candidate_slots=selected_candidate_slots,
        action_classes=4,
        smet_enabled=settings.smet.enabled,
        smet_sparsity=settings.smet.sparsity,
        smet_update_interval=settings.smet.update_interval,
        smet_min_density=settings.smet.min_density,
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
    resume = _restore_temporal_training_state(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        checkpoint_path=resume_checkpoint_path,
        policy=resume_policy,
        reporter=reporter,
    )
    start_step = resume.temporal_step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    metrics = {
        "loss": 0.0,
        "action_loss": 0.0,
        "candidate_loss": 0.0,
        "xy_loss": 0.0,
        "time_loss": 0.0,
    }
    last_committed = start_step
    try:
        for step in range(start_step, max_steps):
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
                    weights=settings.training.temporal_loss_weights,
                )
            if not torch.isfinite(loss.detach()).all():
                raise FloatingPointError("temporal loss became NaN or Inf")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            metrics = {
                "loss": float(loss.detach().cpu()),
                **{
                    key: float(value.detach().cpu())
                    for key, value in loss_parts.items()
                },
            }
            if not math.isfinite(metrics["loss"]):
                raise FloatingPointError("temporal committed loss is not finite")
            last_committed = step + 1
            if should_report_training_step(last_committed, max_steps):
                _report_temporal_step(
                    reporter,
                    step=last_committed,
                    target=max_steps,
                    loss=metrics["loss"],
                    window=window,
                    total_windows=len(source),
                    device=device,
                )
    except BaseException:
        emergency = TemporalTrainingResult(
            run_dir=run_dir,
            checkpoint_path=run_dir / "temporal_emergency.pt",
            device=str(device),
            steps=last_committed,
            windows=len(source),
            sequence_length=selected_sequence_length,
            candidate_slots=selected_candidate_slots,
            input_size=input_size,
            final_loss=metrics["loss"],
            action_loss=metrics["action_loss"],
            candidate_loss=metrics["candidate_loss"],
            xy_loss=metrics["xy_loss"],
            time_loss=metrics["time_loss"],
            loss_weights=settings.training.temporal_loss_weights.model_dump(mode="json"),
            target_strategy=first.target_strategy,
            cuda_max_allocated_gib=None,
            cuda_max_reserved_gib=None,
        )
        _write_checkpoint(
            emergency,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            hidden_size=settings.temporal.hidden_size,
            layers=settings.temporal.layers,
            position=TrainingPosition(
                global_step=last_committed,
                temporal_step=last_committed,
                next_batch_index=last_committed % len(source),
                last_committed_step=last_committed,
            ),
            checkpoint_kind="emergency",
        )
        raise
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
        loss_weights=settings.training.temporal_loss_weights.model_dump(mode="json"),
        target_strategy=first.target_strategy,
        cuda_max_allocated_gib=snapshot.max_allocated_gib,
        cuda_max_reserved_gib=snapshot.max_reserved_gib,
    )
    _write_summary(result)
    _write_checkpoint(
        result,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        hidden_size=settings.temporal.hidden_size,
        layers=settings.temporal.layers,
        position=TrainingPosition(
            global_step=max_steps,
            temporal_step=max_steps,
            next_batch_index=max_steps % len(source),
            last_committed_step=max_steps,
        ),
        checkpoint_kind="latest",
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
    weights,
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
    loss = (
        float(weights.action) * action_loss
        + float(weights.candidate) * candidate_loss
        + float(weights.xy) * xy_loss
        + float(weights.time_offset) * time_loss
    )
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
    optimizer: torch.optim.Optimizer,
    scaler,
    hidden_size: int,
    layers: int,
    position: TrainingPosition,
    checkpoint_kind: str,
) -> None:
    result.run_dir.mkdir(parents=True, exist_ok=True)
    state_source = getattr(model, "_orig_mod", model)
    model_state = state_source.state_dict()
    model_config = {
        "input_size": result.input_size,
        "hidden_size": hidden_size,
        "layers": layers,
        "candidate_slots": result.candidate_slots,
        "action_classes": 4,
    }
    state_source_config = {
        "smet_enabled": any(
            isinstance(module, DynamicSparseLinear)
            for module in state_source.modules()
        )
    }
    if state_source_config["smet_enabled"]:
        first_sparse = next(
            module
            for module in state_source.modules()
            if isinstance(module, DynamicSparseLinear)
        )
        state_source_config.update(
            {
                "smet_sparsity": float(first_sparse.sparsity),
                "smet_update_interval": int(first_sparse.update_interval),
                "smet_min_density": float(first_sparse.min_density),
            }
        )
    model_config.update(state_source_config)
    payload = build_training_checkpoint(
        checkpoint_kind=checkpoint_kind,
        run_id=result.run_dir.name,
        trial_id="temporal",
        models={"temporal": model_state},
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        position=position,
        dataset_state={
            "windows": result.windows,
            "sequence_length": result.sequence_length,
            "candidate_slots": result.candidate_slots,
        },
        resolved_config=model_config,
        extra={
            "legacy_version": "temporal-decision-v1",
            "target_strategy": result.target_strategy,
            "final_loss": result.final_loss,
        },
    )
    payload.update(
        {
            "version": "temporal-decision-v1",
            "model_state": model_state,
            "model_config": model_config,
            "sequence_length": result.sequence_length,
            "target_strategy": result.target_strategy,
        }
    )
    atomic_torch_save_checkpoint(
        payload,
        result.checkpoint_path,
        expected_kind=checkpoint_kind,
    )


def _restore_temporal_training_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    checkpoint_path: Path | None,
    policy: str,
    reporter: TrainingReporter,
) -> TrainingPosition:
    if checkpoint_path is None or policy == "none":
        return TrainingPosition()
    raw = load_training_checkpoint(checkpoint_path)
    model_state = raw.get("model_state")
    if model_state is None and isinstance(raw.get("models"), Mapping):
        model_state = raw["models"].get("temporal")
    if not isinstance(model_state, Mapping):
        raise ValueError("temporal resume checkpoint missing model state")
    strict = policy == "strict"
    restored: list[str] = []
    loaded, skipped = restore_module_state(model, model_state, strict=strict)
    if loaded:
        restored.append("model:temporal")
    if policy != "weights-only":
        if raw.get("optimizer") is not None:
            optimizer.load_state_dict(raw["optimizer"])
            _optimizer_state_to_device(optimizer, next(model.parameters()).device)
            restored.append("optimizer")
        elif strict:
            raise ValueError("strict temporal resume requires optimizer state")
        if raw.get("scaler") is not None:
            scaler.load_state_dict(raw["scaler"])
            restored.append("scaler")
        elif strict:
            raise ValueError("strict temporal resume requires scaler state")
        if restore_rng_state(raw.get("rng_state")):
            restored.append("rng")
        elif strict:
            raise ValueError("strict temporal resume requires rng state")
    position = TrainingPosition.from_mapping(raw.get("training_position"))
    if policy == "weights-only":
        position = TrainingPosition()
    reporter.emit_event(
        TrainingEvent.create(
            event_type="resume.completed",
            severity="success" if not skipped else "warning",
            message_key="inheritance_loaded",
            message_args={"path": str(checkpoint_path)},
            raw_message=(
                "时序训练恢复："
                f"已恢复 {','.join(restored) or '无'}；"
                f"未恢复 {','.join(skipped) or '无'}；"
                f"从 step {position.temporal_step} 继续"
            ),
        )
    )
    return position


def _optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _report_temporal_step(
    reporter: TrainingReporter,
    *,
    step: int,
    target: int,
    loss: float,
    window: TemporalWindow,
    total_windows: int,
    device: torch.device,
) -> None:
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="temporal",
            name="时序训练",
            status="running",
            processed=step,
            total=target,
        )
    )
    reporter.update_metrics(
        temporal_step=step,
        temporal_target=target,
        global_step=step,
        target_global_steps=target,
        loss=loss,
    )
    reporter.report_dataset_usage(
        DatasetUsageState(
            total_segments=total_windows,
            sampled_segments=step,
            unique_segments=min(step, total_windows),
            sampled_frames=step * int(window.frame_mask.sum().item()),
            trained_frames=step * int(window.frame_mask.sum().item()),
            unique_frames=min(step, total_windows),
            total_frames=total_windows,
            duplicate_samples=max(step - total_windows, 0),
            used_sequences=step,
            current_segment=window.sample_keys[-1] if window.sample_keys else None,
        )
    )
    if device.type == "cuda":
        try:
            reporter.report_resource(collect_resource_state())
        except Exception:
            reporter.report_resource(ResourceState())


__all__ = [
    "TemporalTrainingResult",
    "run_temporal_training",
]
