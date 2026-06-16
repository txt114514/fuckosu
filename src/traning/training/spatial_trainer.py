from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from traning.conf import DataSplit, Settings
from traning.core.data_input import build_dataset
from traning.core.memory import (
    CudaRuntimeConfig,
    autocast_context,
    collect_memory_snapshot,
    configure_torch_runtime,
    create_grad_scaler,
    maybe_compile_module,
    module_to_device,
    tensor_to_device,
)
from traning.data import PatchStream, append_color_cues
from traning.models import build_model_stack
from traning.training.losses import compute_spatial_loss
from traning.training.spatial_targets import build_spatial_loss_targets


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
    cuda_max_allocated_gib: float | None
    cuda_max_reserved_gib: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "device": self.device,
            "steps": self.steps,
            "samples_seen": self.samples_seen,
            "last_loss": self.last_loss,
            "last_patch_count": self.last_patch_count,
            "amp_dtype": self.amp_dtype,
            "channels_last": self.channels_last,
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    samples_seen = 0
    last_loss = 0.0
    last_patch_count = 0
    for step in range(max_steps):
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
                    device=device,
                    dtype=torch.float32,
                )
            with autocast_context(device, settings.memory.amp_dtype):
                loss_dict = compute_spatial_loss(prediction, target)
                loss = loss_dict["total"] / planned_patch_count
            scaler.scale(loss).backward()
            accumulated += float(loss_dict["total"].detach().cpu())
            processed += 1
            del patch, local_features, fused, prediction, loss_dict, loss, target

        scaler.step(optimizer)
        scaler.update()
        samples_seen += 1
        last_loss = accumulated / max(processed, 1)
        last_patch_count = processed

    snapshot = collect_memory_snapshot()
    result = SpatialTrainingResult(
        run_dir=run_dir,
        device=str(device),
        steps=max_steps,
        samples_seen=samples_seen,
        last_loss=last_loss,
        last_patch_count=last_patch_count,
        amp_dtype=runtime_state.amp_dtype,
        channels_last=runtime_state.channels_last,
        cuda_max_allocated_gib=snapshot.max_allocated_gib,
        cuda_max_reserved_gib=snapshot.max_reserved_gib,
    )
    _write_summary(result)
    return result


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


__all__ = ["SpatialTrainingResult", "run_spatial_training"]
