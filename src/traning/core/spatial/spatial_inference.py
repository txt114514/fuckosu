from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from traning.lib.data import PatchStream, append_color_cues
from traning.lib.models import build_model_stack
from traning.lib.runtime import (
    CudaRuntimeConfig,
    RuntimeMemoryBudget,
    autocast_context,
    configure_torch_runtime,
    enforce_runtime_memory_budget,
    maybe_compile_module,
    module_to_device,
    tensor_to_device,
)
from traning.lib.training.spatial_decode import (
    SpatialCandidate,
    SpatialPredictionCanvas,
    SpatialPredictionMaps,
    SliderPathCandidate,
    decode_slider_paths,
    decode_spatial_candidates,
)
from traning.conf import Settings


SPATIAL_GPU_TASKS: tuple[str, ...] = (
    "global_encoder_forward",
    "local_encoder_forward_per_patch",
    "global_local_fusion_per_patch",
    "spatial_prediction_head_per_patch",
)

SPATIAL_CPU_TASKS: tuple[str, ...] = (
    "rgb_normalization_and_color_cues",
    "patch_stream_and_padding",
    "prediction_detach_and_canvas_fusion",
    "topk_localmax_nms_candidate_decode",
    "slider_connected_component_path_decode",
    "json_summary_or_candidate_cache_io",
)


@dataclass(frozen=True)
class SpatialFrameInferenceResult:
    candidates: tuple[SpatialCandidate, ...]
    slider_paths: tuple[SliderPathCandidate, ...]
    maps: SpatialPredictionMaps
    memory_budget: RuntimeMemoryBudget
    device: str
    patches_processed: int
    frame_channels: int
    gpu_tasks: tuple[str, ...] = SPATIAL_GPU_TASKS
    cpu_tasks: tuple[str, ...] = SPATIAL_CPU_TASKS

    def as_summary(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "patches_processed": self.patches_processed,
            "frame_channels": self.frame_channels,
            "candidate_count": len(self.candidates),
            "slider_path_count": len(self.slider_paths),
            "ambiguous_slider_path_count": sum(
                1 for path in self.slider_paths if path.ambiguous
            ),
            "ram_budget_gib": self.memory_budget.ram_budget_gib,
            "ram_reserved_for_system_gib": (
                self.memory_budget.ram_reserved_for_system_gib
            ),
            "vram_budget_gib": self.memory_budget.vram_budget_gib,
            "vram_reserved_for_system_gib": (
                self.memory_budget.vram_reserved_for_system_gib
            ),
            "gpu_tasks": self.gpu_tasks,
            "cpu_tasks": self.cpu_tasks,
        }


def run_spatial_frame_inference(
    settings: Settings,
    sample: Mapping[str, Any],
    *,
    device: torch.device,
    max_candidates: int = 16,
    score_threshold: float = 0.0,
    nms_radius_px: float = 32.0,
    slider_threshold: float = 0.5,
    max_slider_paths: int = 16,
    slider_min_cells: int = 4,
    slider_path_points: int = 32,
    patch_limit: int | None = None,
) -> SpatialFrameInferenceResult:
    """Run one-frame spatial inference with explicit GPU/CPU work separation."""

    if patch_limit is not None and patch_limit <= 0:
        raise ValueError("patch_limit must be positive when set")
    memory_budget = enforce_runtime_memory_budget(
        device=device,
        max_vram_gib=settings.memory.max_vram_gib,
        reserve_vram_gib=settings.memory.reserve_vram_gib,
        max_ram_gib=settings.memory.max_ram_gib,
        reserve_ram_gib=settings.memory.reserve_ram_gib,
    )
    frame = _model_frame(sample["image"], settings=settings)
    stream = PatchStream(
        patch_width=settings.tiling.patch_width,
        patch_height=settings.tiling.patch_height,
        overlap_x=settings.tiling.overlap_x,
        overlap_y=settings.tiling.overlap_y,
        pin_memory=settings.loader.pin_memory and device.type == "cuda",
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
    modules = build_model_stack(settings)
    for name, module in tuple(modules.items()):
        moved = module_to_device(
            module,
            device,
            channels_last=runtime_state.channels_last,
        )
        moved = maybe_compile_module(moved, enabled=settings.memory.compile_model)
        moved.eval()
        modules[name] = moved

    frame_device = tensor_to_device(
        frame.unsqueeze(0),
        device,
        channels_last=runtime_state.channels_last,
        non_blocking=stream.pin_memory,
    )
    canvas = SpatialPredictionCanvas(
        frame_width=sample["image"].shape[-1],
        frame_height=sample["image"].shape[-2],
        stride=settings.local_encoder.output_stride,
        embedding_dim=settings.local_encoder.embedding_dim,
    )
    processed = 0
    with torch.no_grad():
        with autocast_context(device, settings.memory.amp_dtype):
            global_features = modules["global"](frame_device)
            global_dense = global_features.dense.detach()
        for patch_cpu, meta in stream.iter_patches(frame):
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
            canvas.write_patch(prediction, meta)
            processed += 1

    maps = canvas.to_maps()
    candidates = decode_spatial_candidates(
        maps,
        max_candidates=max_candidates,
        score_threshold=score_threshold,
        nms_radius_px=nms_radius_px,
    )
    slider_paths = decode_slider_paths(
        maps,
        threshold=slider_threshold,
        min_cells=slider_min_cells,
        max_paths=max_slider_paths,
        sample_points=slider_path_points,
    )
    return SpatialFrameInferenceResult(
        candidates=candidates,
        slider_paths=slider_paths,
        maps=maps,
        memory_budget=memory_budget,
        device=str(device),
        patches_processed=processed,
        frame_channels=frame.shape[0],
    )


def spatial_candidate_to_dict(candidate: SpatialCandidate) -> dict[str, Any]:
    return {
        "x": candidate.x,
        "y": candidate.y,
        "score": candidate.score,
        "object_type": candidate.object_type,
        "object_type_id": candidate.object_type_id,
        "center_score": candidate.center_score,
        "visible_score": candidate.visible_score,
        "type_score": candidate.type_score,
        "ring_score": candidate.ring_score,
        "ring_radius_px": candidate.ring_radius_px,
        "slider_score": candidate.slider_score,
        "spinner_score": candidate.spinner_score,
    }


def slider_path_to_dict(path: SliderPathCandidate) -> dict[str, Any]:
    return {
        "component_id": path.component_id,
        "score": path.score,
        "continuity": path.continuity,
        "ambiguous": path.ambiguous,
        "ambiguity_reasons": path.ambiguity_reasons,
        "bbox": path.bbox,
        "head": path.head,
        "tail": path.tail,
        "polyline": path.polyline,
        "cell_count": path.cell_count,
        "branch_points": path.branch_points,
        "endpoint_count": path.endpoint_count,
    }


def _model_frame(image: torch.Tensor, *, settings: Settings) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("sample image must use CHW layout")
    frame = image.detach().to("cpu")
    if not torch.is_floating_point(frame):
        frame = frame.float().div(255.0)
    return append_color_cues(frame.contiguous(), mode=settings.input.color_cues)


__all__ = [
    "SPATIAL_CPU_TASKS",
    "SPATIAL_GPU_TASKS",
    "SpatialFrameInferenceResult",
    "run_spatial_frame_inference",
    "spatial_candidate_to_dict",
    "slider_path_to_dict",
]
