from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch

from traning.lib.data import PatchStream, append_color_cues
from traning.lib.models import build_model_stack
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
from traning.lib.training.losses import compute_spatial_loss, temporal_consistency_loss
from traning.lib.training.spatial_targets import build_spatial_loss_targets
from traning.conf import DataSplit, Settings
from traning.core.dataset_import import build_dataset
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
class SpatialTrainingResult:
    run_dir: Path
    device: str
    steps: int
    samples_seen: int
    last_loss: float
    last_patch_count: int
    amp_dtype: str
    channels_last: bool
    ram_budget_gib: float
    ram_reserved_for_system_gib: float
    vram_budget_gib: float | None
    vram_reserved_for_system_gib: float | None
    cuda_max_allocated_gib: float | None
    cuda_max_reserved_gib: float | None
    checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_path is None:
            object.__setattr__(self, "checkpoint_path", self.run_dir / "spatial_model.pt")

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "steps": self.steps,
            "samples_seen": self.samples_seen,
            "last_loss": self.last_loss,
            "last_patch_count": self.last_patch_count,
            "amp_dtype": self.amp_dtype,
            "channels_last": self.channels_last,
            "ram_budget_gib": self.ram_budget_gib,
            "ram_reserved_for_system_gib": self.ram_reserved_for_system_gib,
            "vram_budget_gib": self.vram_budget_gib,
            "vram_reserved_for_system_gib": self.vram_reserved_for_system_gib,
            "cuda_max_allocated_gib": self.cuda_max_allocated_gib,
            "cuda_max_reserved_gib": self.cuda_max_reserved_gib,
        }


def run_spatial_training(
    settings: Settings,
    *,
    device: torch.device,
    run_dir: Path,
    split: DataSplit = "train",
    max_steps: int = 1,
    learning_rate: float = 1e-4,
    patch_limit: int | None = None,
    dataset: Sequence[dict[str, Any]] | None = None,
    reporter: TrainingReporter = NullReporter(),
    resume_checkpoint_path: Path | None = None,
    resume_policy: str = "none",
) -> SpatialTrainingResult:
    """Run the first-version single-frame spatial training loop."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if patch_limit is not None and patch_limit <= 0:
        raise ValueError("patch_limit must be positive when set")

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(settings.runtime.seed)
    memory_budget = enforce_runtime_memory_budget(
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
            channels_last=settings.memory.channels_last,
        ),
    )
    stream = PatchStream(
        patch_width=settings.tiling.patch_width,
        patch_height=settings.tiling.patch_height,
        overlap_x=settings.tiling.overlap_x,
        overlap_y=settings.tiling.overlap_y,
        pin_memory=settings.loader.pin_memory and device.type == "cuda",
    )
    training_dataset = (
        dataset if dataset is not None else build_dataset(settings, split=split)
    )
    if len(training_dataset) == 0:
        raise ValueError("training dataset must not be empty")

    modules = build_model_stack(settings)
    for frozen_name in ("global", "structure"):
        for parameter in modules[frozen_name].parameters():
            parameter.requires_grad_(False)
    for name, module in tuple(modules.items()):
        moved = module_to_device(
            module,
            device,
            channels_last=runtime_state.channels_last,
        )
        moved = maybe_compile_module(
            moved,
            enabled=settings.memory.compile_model,
        )
        moved.train(name not in {"global", "structure"})
        modules[name] = moved

    parameters = [
        parameter
        for module in modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    scaler = create_grad_scaler(
        device=device,
        amp_dtype=settings.memory.amp_dtype,
        mode=settings.memory.grad_scaler,
    )
    resume = _restore_spatial_training_state(
        modules=modules,
        optimizer=optimizer,
        scaler=scaler,
        checkpoint_path=resume_checkpoint_path,
        policy=resume_policy,
        reporter=reporter,
    )
    start_step = resume.spatial_step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    samples_seen = 0
    last_loss = 0.0
    last_patch_count = 0
    last_committed = start_step
    try:
        for step in range(start_step, max_steps):
            sample = training_dataset[step % len(training_dataset)]
            frame_cpu = _normalize_frame(sample["image"])
            model_frame_cpu = append_color_cues(
                frame_cpu,
                mode=settings.input.color_cues,
            )
            optimizer.zero_grad(set_to_none=True)
            frame = tensor_to_device(
                model_frame_cpu.unsqueeze(0),
                device,
                channels_last=runtime_state.channels_last,
                non_blocking=stream.pin_memory,
            )
            with torch.no_grad():
                with autocast_context(device, settings.memory.amp_dtype):
                    global_features = modules["global"](frame)
                    global_dense = global_features.dense.detach()

            patch_total = stream.count(model_frame_cpu)
            planned_patch_count = min(patch_total, patch_limit or patch_total)
            accumulated = 0.0
            processed = 0
            for patch_cpu, meta in stream.iter_patches(model_frame_cpu):
                if patch_limit is not None and processed >= patch_limit:
                    break
                patch = tensor_to_device(
                    patch_cpu.unsqueeze(0),
                    device,
                    channels_last=runtime_state.channels_last,
                    non_blocking=stream.pin_memory,
                )
                with autocast_context(device, settings.memory.amp_dtype):
                    local_features = modules["local"](patch)
                    fused = modules["fusion"](
                        local_features=local_features,
                        global_features=global_dense,
                        patch_meta=meta,
                    )
                    prediction = modules["head"](fused.dense)
                with torch.no_grad():
                    target = build_spatial_loss_targets(
                        sample,
                        meta,
                        prediction.center_heatmap.shape[-2:],
                        settings=settings,
                        device=device,
                        dtype=torch.float32,
                    )
                with autocast_context(device, settings.memory.amp_dtype):
                    loss_dict = compute_spatial_loss(prediction, target)
                    _add_spatial_consistency_losses(
                        loss_dict,
                        prediction=prediction,
                        target=target,
                        weights=settings.training.spatial_consistency_loss_weights,
                    )
                    loss = loss_dict["total"] / planned_patch_count
                if not torch.isfinite(loss.detach()).all():
                    raise FloatingPointError("spatial loss became NaN or Inf")
                scaler.scale(loss).backward()
                accumulated += float(loss_dict["total"].detach().cpu())
                processed += 1
                del patch, local_features, fused, prediction, loss_dict, loss, target

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            samples_seen += 1
            last_loss = accumulated / max(processed, 1)
            if not math.isfinite(last_loss):
                raise FloatingPointError("spatial committed loss is not finite")
            last_patch_count = processed
            last_committed = step + 1
            if should_report_training_step(last_committed, max_steps):
                _report_spatial_step(
                    reporter,
                    step=last_committed,
                    target=max_steps,
                    loss=last_loss,
                    sample=sample,
                    total_samples=len(training_dataset),
                    generated_patches=processed,
                    device=device,
                )
    except BaseException:
        _write_checkpoint(
            SpatialTrainingResult(
                run_dir=run_dir,
                checkpoint_path=run_dir / "spatial_emergency.pt",
                device=str(device),
                steps=last_committed,
                samples_seen=samples_seen,
                last_loss=last_loss,
                last_patch_count=last_patch_count,
                amp_dtype=runtime_state.amp_dtype,
                channels_last=runtime_state.channels_last,
                ram_budget_gib=memory_budget.ram_budget_gib,
                ram_reserved_for_system_gib=memory_budget.ram_reserved_for_system_gib,
                vram_budget_gib=memory_budget.vram_budget_gib,
                vram_reserved_for_system_gib=memory_budget.vram_reserved_for_system_gib,
                cuda_max_allocated_gib=None,
                cuda_max_reserved_gib=None,
            ),
            modules=modules,
            settings=settings,
            optimizer=optimizer,
            scaler=scaler,
            position=TrainingPosition(
                global_step=last_committed,
                spatial_step=last_committed,
                next_batch_index=last_committed % len(training_dataset),
                last_committed_step=last_committed,
            ),
            checkpoint_kind="emergency",
        )
        raise

    snapshot = collect_memory_snapshot()
    result = SpatialTrainingResult(
        run_dir=run_dir,
        checkpoint_path=run_dir / "spatial_model.pt",
        device=str(device),
        steps=max_steps,
        samples_seen=samples_seen,
        last_loss=last_loss,
        last_patch_count=last_patch_count,
        amp_dtype=runtime_state.amp_dtype,
        channels_last=runtime_state.channels_last,
        ram_budget_gib=memory_budget.ram_budget_gib,
        ram_reserved_for_system_gib=memory_budget.ram_reserved_for_system_gib,
        vram_budget_gib=memory_budget.vram_budget_gib,
        vram_reserved_for_system_gib=memory_budget.vram_reserved_for_system_gib,
        cuda_max_allocated_gib=snapshot.max_allocated_gib,
        cuda_max_reserved_gib=snapshot.max_reserved_gib,
    )
    _write_summary(result)
    _write_checkpoint(
        result,
        modules=modules,
        settings=settings,
        optimizer=optimizer,
        scaler=scaler,
        position=TrainingPosition(
            global_step=max_steps,
            spatial_step=max_steps,
            next_batch_index=max_steps % len(training_dataset),
            last_committed_step=max_steps,
        ),
        checkpoint_kind="latest",
    )
    return result


def _add_spatial_consistency_losses(
    loss_dict: dict[str, torch.Tensor],
    *,
    prediction,
    target,
    weights,
) -> None:
    total = loss_dict["total"]
    if weights.embedding > 0:
        embedding = prediction.candidate_embedding
        embedding_loss = embedding.var(dim=(-2, -1), unbiased=False).mean()
        loss_dict["embedding_consistency"] = embedding_loss
        total = total + float(weights.embedding) * embedding_loss
    if weights.ring_radius > 0:
        ring_loss = temporal_consistency_loss(
            prediction.ring_radius[:, :, :, 1:],
            prediction.ring_radius[:, :, :, :-1],
            mask=(target.ring_mask[:, :, :, 1:] > 0).to(prediction.ring_radius.dtype),
        )
        loss_dict["ring_radius_consistency"] = ring_loss
        total = total + float(weights.ring_radius) * ring_loss
    if weights.slider_continuity > 0:
        slider_loss = temporal_consistency_loss(
            prediction.slider_direction[:, :, :, 1:],
            prediction.slider_direction[:, :, :, :-1],
            mask=(target.slider_mask[:, :, :, 1:] > 0).to(
                prediction.slider_direction.dtype
            ),
        )
        loss_dict["slider_continuity"] = slider_loss
        total = total + float(weights.slider_continuity) * slider_loss
    loss_dict["total"] = total


def _normalize_frame(frame: torch.Tensor) -> torch.Tensor:
    if frame.ndim != 3:
        raise ValueError("sample image must use CHW layout")
    if not torch.is_floating_point(frame):
        frame = frame.float().div(255.0)
    return frame.contiguous()


def _write_summary(result: SpatialTrainingResult) -> None:
    result.run_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {value}" for key, value in result.as_dict().items()]
    (result.run_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_checkpoint(
    result: SpatialTrainingResult,
    *,
    modules: dict[str, torch.nn.Module],
    settings: Settings,
    optimizer: torch.optim.Optimizer,
    scaler,
    position: TrainingPosition,
    checkpoint_kind: str,
) -> None:
    result.run_dir.mkdir(parents=True, exist_ok=True)
    model_state = {
        name: getattr(module, "_orig_mod", module).state_dict()
        for name, module in modules.items()
    }
    model_config = {
        "input": settings.input.model_dump(mode="json"),
        "tiling": settings.tiling.model_dump(mode="json"),
        "local_encoder": settings.local_encoder.model_dump(mode="json"),
        "global_encoder": settings.global_encoder.model_dump(mode="json"),
        "fusion": settings.fusion.model_dump(mode="json"),
    }
    payload = build_training_checkpoint(
        checkpoint_kind=checkpoint_kind,
        run_id=result.run_dir.name,
        trial_id="spatial",
        models=model_state,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        position=position,
        dataset_state={
            "samples_seen": result.samples_seen,
            "last_patch_count": result.last_patch_count,
        },
        resolved_config=model_config,
        dataset_fingerprint={
            "dataset_root": str(settings.data_input.dataset_root),
            "split_manifest_path": str(settings.data_input.split_manifest_path),
            "train_items": tuple(settings.data_input.train_items),
        },
        extra={
            "legacy_version": "spatial-model-v1",
            "last_loss": result.last_loss,
        },
    )
    payload.update(
        {
            "version": "spatial-model-v1",
            "model_state": model_state,
            "model_config": model_config,
            "steps": result.steps,
            "samples_seen": result.samples_seen,
            "last_loss": result.last_loss,
        }
    )
    atomic_torch_save_checkpoint(
        payload,
        result.checkpoint_path,
        expected_kind=checkpoint_kind,
    )


def _restore_spatial_training_state(
    *,
    modules: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler,
    checkpoint_path: Path | None,
    policy: str,
    reporter: TrainingReporter,
) -> TrainingPosition:
    if checkpoint_path is None or policy == "none":
        return TrainingPosition()
    raw = load_training_checkpoint(checkpoint_path)
    model_state = raw.get("models") or raw.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("spatial resume checkpoint missing model state")
    strict = policy == "strict"
    restored: list[str] = []
    skipped: list[str] = []
    for name, module in modules.items():
        state = model_state.get(name)
        if isinstance(state, dict):
            loaded, missing = restore_module_state(module, state, strict=strict)
            if loaded:
                restored.append(f"model:{name}")
            skipped.extend(f"model:{name}:{item}" for item in missing)
    if policy != "weights-only":
        if raw.get("optimizer") is not None:
            optimizer.load_state_dict(raw["optimizer"])
            _optimizer_state_to_device(
                optimizer,
                next(modules["local"].parameters()).device,
            )
            restored.append("optimizer")
        elif strict:
            raise ValueError("strict spatial resume requires optimizer state")
        if raw.get("scaler") is not None:
            scaler.load_state_dict(raw["scaler"])
            restored.append("scaler")
        elif strict:
            raise ValueError("strict spatial resume requires scaler state")
        if restore_rng_state(raw.get("rng_state")):
            restored.append("rng")
        elif strict:
            raise ValueError("strict spatial resume requires rng state")
    position = TrainingPosition.from_mapping(raw.get("training_position"))
    if policy == "weights-only":
        position = TrainingPosition()
    elif position.spatial_step == 0 and raw.get("steps") is not None:
        position = TrainingPosition(
            global_step=int(raw.get("steps", 0)),
            spatial_step=int(raw.get("steps", 0)),
            last_committed_step=int(raw.get("steps", 0)),
        )
    reporter.emit_event(
        TrainingEvent.create(
            event_type="resume.completed",
            severity="success" if not skipped else "warning",
            message_key="inheritance_loaded",
            message_args={"path": str(checkpoint_path)},
            raw_message=(
                "空间训练恢复："
                f"已恢复 {','.join(restored) or '无'}；"
                f"未恢复 {','.join(skipped) or '无'}；"
                f"从 step {position.spatial_step} 继续"
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


def _report_spatial_step(
    reporter: TrainingReporter,
    *,
    step: int,
    target: int,
    loss: float,
    sample: dict[str, Any],
    total_samples: int,
    generated_patches: int,
    device: torch.device,
) -> None:
    reporter.update_pipeline_stage(
        PipelineStageState(
            stage_id="spatial",
            name="空间训练",
            status="running",
            processed=step,
            total=target,
        )
    )
    reporter.update_metrics(
        spatial_step=step,
        spatial_target=target,
        global_step=step,
        target_global_steps=target,
        loss=loss,
    )
    reporter.report_dataset_usage(
        DatasetUsageState(
            total_segments=total_samples,
            sampled_segments=step,
            unique_segments=min(step, total_samples),
            sampled_frames=step,
            trained_frames=step,
            unique_frames=min(step, total_samples),
            total_frames=total_samples,
            duplicate_samples=max(step - total_samples, 0),
            generated_patches=generated_patches,
            current_segment=str(sample.get("sample_key")),
        )
    )
    if device.type == "cuda":
        try:
            reporter.report_resource(collect_resource_state())
        except Exception:
            reporter.report_resource(ResourceState())


__all__ = ["SpatialTrainingResult", "run_spatial_training"]
