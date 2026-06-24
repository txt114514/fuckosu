from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from traning.conf import Settings
from traning.core.temporal import (
    ACTION_NAMES,
    TemporalCandidateWindowDataset,
    TemporalWindow,
)
from traning.lib.models import CausalTemporalModel
from traning.lib.runtime import (
    CudaRuntimeConfig,
    autocast_context,
    configure_torch_runtime,
    enforce_runtime_memory_budget,
    module_to_device,
    tensor_to_device,
)


DECISION_OUTPUT_VERSION = "temporal-decision-v1"


@dataclass(frozen=True)
class TemporalDecisionRunResult:
    output_dir: Path
    manifest_path: Path
    decisions_path: Path
    checkpoint_path: Path
    device: str
    frames: int
    sequence_length: int
    candidate_slots: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "decisions_path": self.decisions_path,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "frames": self.frames,
            "sequence_length": self.sequence_length,
            "candidate_slots": self.candidate_slots,
        }


def run_temporal_decision(
    settings: Settings,
    *,
    cache_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> TemporalDecisionRunResult:
    checkpoint = _load_checkpoint(checkpoint_path, device=device)
    model_config = checkpoint["model_config"]
    sequence_length = int(checkpoint["sequence_length"])
    candidate_slots = int(model_config["candidate_slots"])
    dataset = TemporalCandidateWindowDataset.from_cache_dir(
        cache_dir,
        sequence_length=sequence_length,
        candidate_slots=candidate_slots,
    )
    model = CausalTemporalModel(**model_config)
    model.load_state_dict(checkpoint["model_state"])
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
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "decisions.jsonl"
    frames = 0
    with decisions_path.open("w", encoding="utf-8") as handle:
        with torch.no_grad(), autocast_context(device, runtime_state.amp_dtype):
            for window in dataset:
                batch = tensor_to_device(
                    window.features.unsqueeze(1),
                    device,
                    channels_last=False,
                    non_blocking=False,
                )
                outputs, _ = model(batch)
                for frame_index, output in enumerate(outputs):
                    if not bool(window.frame_mask[frame_index]):
                        continue
                    row = _decision_row(window, frame_index, output)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    frames += 1
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "version": DECISION_OUTPUT_VERSION,
        "frames": frames,
        "checkpoint": str(checkpoint_path),
        "candidate_cache": str(cache_dir),
        "decisions": decisions_path.name,
        "device": str(device),
        "sequence_length": sequence_length,
        "candidate_slots": candidate_slots,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return TemporalDecisionRunResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        decisions_path=decisions_path,
        checkpoint_path=checkpoint_path,
        device=str(device),
        frames=frames,
        sequence_length=sequence_length,
        candidate_slots=candidate_slots,
    )


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("temporal checkpoint must be a mapping")
    if checkpoint.get("version") != DECISION_OUTPUT_VERSION:
        raise ValueError(
            f"unsupported temporal checkpoint version: {checkpoint.get('version')!r}"
        )
    if not isinstance(checkpoint.get("model_config"), Mapping):
        raise ValueError("temporal checkpoint missing model_config")
    if "model_state" not in checkpoint:
        raise ValueError("temporal checkpoint missing model_state")
    return checkpoint


def _decision_row(
    window: TemporalWindow,
    frame_index: int,
    output,
) -> dict[str, Any]:
    action_probs = F.softmax(output.action_logits[0], dim=0).detach().cpu()
    action_id = int(action_probs.argmax().item())
    candidate_logits = output.selected_candidate_logits[0].detach().cpu().clone()
    mask = window.candidate_mask[frame_index]
    if bool(mask.any()):
        candidate_logits[~mask] = float("-inf")
        selected_slot = int(candidate_logits.argmax().item())
        selected_score = float(F.softmax(candidate_logits, dim=0)[selected_slot].item())
        selected_candidate_id = window.candidate_ids[frame_index][selected_slot]
        selected_candidate_xy = [
            float(window.candidate_features[frame_index, selected_slot, 1].item()),
            float(window.candidate_features[frame_index, selected_slot, 2].item()),
        ]
    else:
        selected_slot = None
        selected_score = None
        selected_candidate_id = None
        selected_candidate_xy = None
    return {
        "version": DECISION_OUTPUT_VERSION,
        "sample_key": window.sample_keys[frame_index],
        "frame_index": window.frame_indices[frame_index],
        "timestamp_ms": window.timestamps_ms[frame_index],
        "action": ACTION_NAMES[action_id],
        "action_id": action_id,
        "action_probability": float(action_probs[action_id].item()),
        "selected_candidate_slot": selected_slot,
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_probability": selected_score,
        "selected_candidate_xy_normalized": selected_candidate_xy,
        "predicted_xy_normalized": [
            float(output.x[0, 0].detach().cpu().item()),
            float(output.y[0, 0].detach().cpu().item()),
        ],
        "time_offset_ms": float(output.time_offset_ms[0, 0].detach().cpu().item()),
    }


__all__ = [
    "DECISION_OUTPUT_VERSION",
    "TemporalDecisionRunResult",
    "run_temporal_decision",
]
