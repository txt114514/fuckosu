from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from PIL import Image, ImageDraw
import typer
from rich.console import Console
from rich.table import Table

from traning.lib.data import build_patch_windows
from traning.lib.data import PatchStream, append_color_cues
from traning.lib.models import build_model_stack
from traning.lib.runtime import (
    CudaRuntimeConfig,
    autocast_context,
    collect_memory_snapshot,
    configure_torch_runtime,
    create_grad_scaler,
    enforce_runtime_memory_budget,
    format_oom_guidance,
    maybe_compile_module,
    module_to_device,
    tensor_to_device,
)
from traning.conf import DataSplit, load_settings
from traning.core.decision import (
    FullTrainingRunConfig,
    generate_candidate_cache,
    run_full_training_pipeline,
    run_temporal_decision,
)
from traning.core.dataset_import import build_dataset, inspect_data_input
from traning.core.spatial import (
    run_spatial_frame_inference,
    run_spatial_training,
    slider_path_to_dict,
    spatial_candidate_to_dict,
)
from traning.core.temporal import run_temporal_training
from traning.core.result_export import (
    save_annotation_gallery,
    visualize_click_label,
)
from traning.core.training_ramp import run_training_ramp
from traning.core.full_flow import (
    DEFAULT_FULL_FLOW_ROOT,
    FullFlowConfig,
    load_full_flow_status,
    run_full_flow,
)
from traning.core.training_inheritance import (
    create_inheritance_package,
    load_inheritance_package,
)
from traning.state import load_batch_gallery_request
from environment import collect_environment_report
from visualization.lib import (
    TrainingEvent,
    create_dashboard_reporter,
)


app = typer.Typer(help="osu! video model training commands.")
console = Console()
DEFAULT_TRAINING_CONFIG = Path("configs/model_small_vram.yaml")


class CliParameterError(ValueError):
    """Raised when a plain business entry receives an invalid CLI-like value."""


def _render_report(report) -> None:
    table = Table(title=f"Data input ({report.split})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("segments", str(report.segment_count))
    table.add_row("estimated frames", str(report.frame_count_estimate))
    table.add_row("issues", str(report.issue_count))
    table.add_row("items", str(report.item_counts))
    table.add_row("dimensions", str(report.dimension_counts))
    table.add_row("categories", str(report.category_counts))
    console.print(table)
    for issue in report.issues[:20]:
        console.print(f"[red]- {issue}[/red]")


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _format_gib(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f} GiB"


def _render_env_report(report) -> None:
    torch_table = Table(title="Training environment")
    torch_table.add_column("Metric")
    torch_table.add_column("Value")
    torch_table.add_row("python", report.python_version)
    torch_table.add_row("executable", report.python_executable)
    torch_table.add_row("platform", report.platform)
    torch_table.add_row("ffmpeg", report.ffmpeg_path or "missing")
    torch_table.add_row("nvidia-smi", report.nvidia_smi_path or "missing")
    torch_table.add_row("torch", report.torch.version or "missing")
    torch_table.add_row("torchvision", report.torch.torchvision_version or "missing")
    torch_table.add_row("torch cuda", report.torch.torch_cuda or "missing")
    torch_table.add_row("cuda available", _format_bool(report.torch.cuda_available))
    torch_table.add_row("gpu", report.torch.gpu_name or "unavailable")
    torch_table.add_row(
        "compute capability", report.torch.compute_capability or "unknown"
    )
    torch_table.add_row("cuDNN", report.torch.cudnn_version or "unknown")
    torch_table.add_row("bf16", _format_bool(report.torch.bf16_supported))
    torch_table.add_row("free vram", _format_gib(report.torch.free_vram_gib))
    torch_table.add_row("total vram", _format_gib(report.torch.total_vram_gib))
    if report.torch.error:
        torch_table.add_row("torch warning", report.torch.error)
    console.print(torch_table)

    package_table = Table(title="Python packages")
    package_table.add_column("Package")
    package_table.add_column("Required")
    package_table.add_column("Import")
    package_table.add_column("Version")
    for check in report.packages:
        package_table.add_row(
            check.spec.label,
            _format_bool(check.spec.required),
            _format_bool(check.available),
            check.version or "unknown",
        )
    console.print(package_table)


def _run_dir(kind: str, *, root: Path | None = None) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = (root or Path("runs")) / f"{run_id}__{kind}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _select_device(device: str) -> torch.device:
    if device == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected = device
    if selected == "cuda" and not torch.cuda.is_available():
        raise CliParameterError("CUDA is not available inside the container")
    if selected not in {"cpu", "cuda"}:
        raise CliParameterError("device must be cpu, cuda, or auto")
    return torch.device(selected)


def _load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    data = torch.from_numpy(np.asarray(image).copy())
    return data.permute(2, 0, 1).float().div(255.0)


def _build_model_stack(settings) -> dict[str, torch.nn.Module]:
    return build_model_stack(settings)


def _execute_model_smoke(
    *,
    config: Path | None,
    device: torch.device,
    backward: bool,
) -> dict[str, Any]:
    settings = load_settings(config)
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
    frame = torch.rand(
        3,
        settings.input.height,
        settings.input.width,
        dtype=torch.float32,
    )
    frame = append_color_cues(frame, mode=settings.input.color_cues)
    patch, meta = next(stream.iter_patches(frame))
    patch_count = stream.count(frame)
    frame = tensor_to_device(
        frame.unsqueeze(0),
        device,
        channels_last=runtime_state.channels_last,
        non_blocking=stream.pin_memory,
    )
    patch = tensor_to_device(
        patch.unsqueeze(0),
        device,
        channels_last=runtime_state.channels_last,
        non_blocking=stream.pin_memory,
    )
    modules = _build_model_stack(settings)
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
        moved.train(backward)
        modules[name] = moved
    parameters = [
        parameter
        for module in modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4) if backward else None
    scaler = create_grad_scaler(
        device=device,
        amp_dtype=settings.memory.amp_dtype,
        mode=settings.memory.grad_scaler,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, settings.memory.amp_dtype):
            global_features = modules["global"](frame)
            global_prediction = modules["structure"](global_features.dense)
            local_features = modules["local"](patch)
            fused = modules["fusion"](
                local_features=local_features,
                global_features=global_features.dense,
                patch_meta=meta,
            )
            spatial = modules["head"](fused.dense)
            loss = (
                spatial.center_heatmap.mean()
                + spatial.visible_heatmap.mean()
                + global_prediction.objectness.mean()
            )
        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            console.print(
                "[red]"
                + format_oom_guidance(
                    patch_size=(
                        settings.tiling.patch_width,
                        settings.tiling.patch_height,
                    ),
                    global_size=(
                        settings.global_encoder.input_width,
                        settings.global_encoder.input_height,
                    ),
                    batch_size=1,
                    amp_dtype=settings.memory.amp_dtype,
                    config_path=str(config) if config is not None else None,
                )
                + "[/red]"
            )
        raise
    snapshot = collect_memory_snapshot()
    return {
        "device": str(device),
        "amp_dtype": runtime_state.amp_dtype,
        "channels_last": runtime_state.channels_last,
        "allow_tf32": runtime_state.allow_tf32,
        "cudnn_benchmark": runtime_state.cudnn_benchmark,
        "matmul_precision": runtime_state.matmul_float32_precision,
        "grad_scaler": scaler.is_enabled(),
        "compile_model": settings.memory.compile_model,
        "ram_budget_gib": memory_budget.ram_budget_gib,
        "ram_reserved_for_system_gib": memory_budget.ram_reserved_for_system_gib,
        "vram_budget_gib": memory_budget.vram_budget_gib,
        "vram_reserved_for_system_gib": memory_budget.vram_reserved_for_system_gib,
        "cuda_memory_fraction": memory_budget.cuda_memory_fraction,
        "patch_count": patch_count,
        "patch_shape": tuple(patch.shape),
        "local_shape": tuple(local_features.dense.shape),
        "global_shape": tuple(global_features.dense.shape),
        "fused_shape": tuple(fused.dense.shape),
        "loss": float(loss.detach().cpu()),
        "cuda_max_allocated_gib": snapshot.max_allocated_gib,
        "cuda_max_reserved_gib": snapshot.max_reserved_gib,
    }


def _render_dict_table(title: str, values: dict[str, Any]) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in values.items():
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        table.add_row(key, rendered)
    console.print(table)


def _render_parameter_group_score(evaluation) -> None:
    table = Table(title="Parameter group score")
    table.add_column("Group")
    table.add_column("Quality")
    table.add_column("Passed")
    table.add_column("Targets")
    table.add_column("Hits")
    table.add_column("Misses")
    table.add_column("Unresolved")
    table.add_column("Freq limited")
    table.add_column("No-op")
    table.add_column("Actions")
    table.add_column("Gallery")
    table.add_row(
        str(evaluation.parameter_group_id),
        f"{evaluation.quality_score:.6f}",
        "yes" if evaluation.passed else "no",
        str(evaluation.target_count),
        str(evaluation.hit_count),
        str(evaluation.miss_count),
        str(evaluation.unresolved_count),
        str(evaluation.frequency_limited_count),
        str(evaluation.no_op_frame_count),
        str(evaluation.action_frame_count),
        (
            f"{evaluation.gallery_status}: {evaluation.gallery_output_dir}"
            if evaluation.gallery_output_dir is not None
            else evaluation.gallery_status
        ),
    )
    console.print(table)
    if evaluation.gallery_warning:
        console.print(f"[yellow]{evaluation.gallery_warning}[/yellow]")


def _compact_slider_path(path: dict[str, Any]) -> dict[str, Any]:
    return {
        "component_id": path["component_id"],
        "score": path["score"],
        "continuity": path["continuity"],
        "ambiguous": path["ambiguous"],
        "ambiguity_reasons": path["ambiguity_reasons"],
        "bbox": path["bbox"],
        "head": path["head"],
        "tail": path["tail"],
        "cell_count": path["cell_count"],
        "branch_points": path["branch_points"],
        "endpoint_count": path["endpoint_count"],
    }


def _write_summary_txt(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )


def _write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def inspect_training_data(
    *,
    config: Path | None = None,
    split: DataSplit = "all",
):
    settings = load_settings(config)
    report = inspect_data_input(settings, split=split)
    return settings, report


def collect_training_environment():
    return collect_environment_report()


def preview_training_sample(
    *,
    index: int = 0,
    split: DataSplit = "train",
    config: Path | None = None,
) -> dict[str, Any]:
    settings = load_settings(config)
    dataset = build_dataset(settings, split=split)
    if index >= len(dataset):
        raise CliParameterError(
            f"index {index} is outside dataset length {len(dataset)}"
        )
    sample = dataset[index]
    _, height, width = sample["image"].shape
    data = settings.data_input
    windows = build_patch_windows(
        width,
        height,
        patch_width=data.patch_width,
        patch_height=data.patch_height,
        overlap_x=data.overlap_x,
        overlap_y=data.overlap_y,
    )
    return {
        "sample_key": sample["sample_key"],
        "frame_index": sample["frame_index"],
        "timestamp_ms": sample["timestamp_ms"],
        "image_shape": tuple(sample["image"].shape),
        "visible_objects": len(sample["visible_hit_objects"]),
        "patch_count": len(windows),
        "first_patch": windows[0],
    }


def run_training(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    split: DataSplit = "train",
    device: str = "auto",
    spatial_max_steps: int = 1,
    temporal_max_steps: int = 1,
    spatial_learning_rate: float = 1e-4,
    temporal_learning_rate: float = 1e-4,
    patch_limit: int = 1,
    cache_max_frames: int = 1,
    sequence_length: int | None = None,
    candidate_slots: int | None = None,
    parameter_group_id: str = "pg-0001",
    render_gallery: bool = True,
    gallery_output_root: Path | None = None,
    gallery_samples_per_group: int | None = None,
    progress_ui: str = "auto",
    progress_language: str = "zh-CN",
    inherit_from: Path | str | None = None,
    resume_policy: str = "none",
):
    if progress_ui not in {"auto", "rich", "plain", "off"}:
        raise CliParameterError("progress-ui must be auto, rich, plain, or off")
    if resume_policy not in {"strict", "auto", "weights-only", "none"}:
        raise CliParameterError("resume-policy must be strict, auto, weights-only, or none")
    settings = load_settings(config)
    selected = _select_device(device)
    run_dir = _run_dir("full_training")
    with create_dashboard_reporter(
        run_id=run_dir.name,
        output_dir=run_dir / "dashboard",
        progress_ui=progress_ui,
        progress_language=progress_language,
    ) as dashboard:
        reporter = dashboard.reporter
        inheritance = load_inheritance_package(
            inherit_from=inherit_from,
            current_settings=settings,
            policy=resume_policy,  # type: ignore[arg-type]
        )
        _write_json_report(run_dir / "dashboard" / "resume_report.json", inheritance.as_dict())
        if inheritance.status not in {"skipped", "missing"}:
            reporter.emit_event(
                TrainingEvent.create(
                    event_type="inheritance",
                    severity="success" if inheritance.compatible else "warning",
                    message_key=(
                        "inheritance_loaded"
                        if inheritance.compatible
                        else "inheritance_downgraded"
                    ),
                    message_args={
                        "path": str(inheritance.path),
                        "reason": ",".join(inheritance.downgrade_reasons),
                    },
                )
            )
        result = run_full_training_pipeline(
            settings,
            config=FullTrainingRunConfig(
                run_dir=run_dir,
                device=selected,
                split=split,
                spatial_max_steps=spatial_max_steps,
                temporal_max_steps=temporal_max_steps,
                spatial_learning_rate=spatial_learning_rate,
                temporal_learning_rate=temporal_learning_rate,
                patch_limit=None if patch_limit == 0 else patch_limit,
                cache_max_frames=(
                    None if cache_max_frames == 0 else cache_max_frames
                ),
                sequence_length=sequence_length,
                candidate_slots=candidate_slots,
                parameter_group_id=parameter_group_id,
                render_gallery=render_gallery,
                gallery_output_root=gallery_output_root,
                gallery_samples_per_group=gallery_samples_per_group,
                reporter=reporter,
                resume_policy=inheritance.policy,
                resume_stage_checkpoints=inheritance.stage_checkpoint_paths,
            ),
        )
        package = _safe_create_inheritance_package(
            run_dir=run_dir,
            settings=settings,
            config=config,
            result=result,
            reporter=reporter,
        )
        if package is not None:
            reporter.emit_event(
                TrainingEvent.create(
                    event_type="inheritance",
                    severity="success",
                    message_key="inheritance_saved",
                    message_args={"path": str(package.path)},
                )
            )
        return result


def _safe_create_inheritance_package(
    *,
    run_dir: Path,
    settings,
    config: Path,
    result,
    reporter,
):
    try:
        return create_inheritance_package(
            output_dir=run_dir,
            settings=settings,
            resolved_config_path=config,
            latest_checkpoint_path=result.temporal.checkpoint_path,
            best_checkpoint_path=result.temporal.checkpoint_path,
            stage_checkpoints={
                "spatial": result.spatial.checkpoint_path,
                "temporal": result.temporal.checkpoint_path,
            },
            training_state=result.as_summary(),
            score_state=result.evaluation.as_dict(),
            artifacts={
                "gallery": result.evaluation.gallery_output_dir,
                "score": result.evaluation.report_path,
                "next_job": result.evaluation.next_job_path,
            },
        )
    except Exception as error:
        reporter.emit_event(
            TrainingEvent.create(
                event_type="inheritance",
                severity="warning",
                message_key="fatal_error",
                message_args={"error": f"继承包生成失败：{error}"},
            )
        )
        return None


def run_training_job_spec(
    *,
    job: Path,
    config: Path = DEFAULT_TRAINING_CONFIG,
    device: str = "auto",
    execute: bool = True,
):
    raw = json.loads(job.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CliParameterError("training job spec must be a JSON object")
    trial_id = str(raw.get("trial_id") or "")
    if not trial_id:
        raise CliParameterError("training job spec is missing trial_id")
    budget_steps = int(raw.get("budget_steps") or 1)
    if budget_steps <= 0:
        raise CliParameterError("training job budget_steps must be positive")
    summary = {
        "job": job,
        "trial_id": trial_id,
        "budget_steps": budget_steps,
        "curriculum_stage": raw.get("curriculum_stage"),
        "rung": raw.get("rung"),
        "parent_checkpoint_path": raw.get("parent_checkpoint_path"),
        "execute": execute,
    }
    if not execute:
        return summary
    result = run_training(
        config=config,
        device=device,
        spatial_max_steps=budget_steps,
        temporal_max_steps=budget_steps,
        parameter_group_id=trial_id,
    )
    return summary | {"result": result.as_summary()}


def run_training_ramp_job(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    device: str = "auto",
    output_root: Path = Path("artifacts") / "training_ramp",
    target_config: Path | None = None,
    run_id: str | None = None,
    auto_launch_full: bool = False,
    force_level: bool = False,
    max_levels: int | None = None,
    run_full_checks: bool = True,
    progress_ui: str = "auto",
    progress_language: str = "zh-CN",
    inherit_from: Path | str | None = None,
    resume_policy: str = "none",
):
    if progress_ui not in {"auto", "rich", "plain", "off"}:
        raise CliParameterError("progress-ui must be auto, rich, plain, or off")
    if resume_policy not in {"strict", "auto", "weights-only", "none"}:
        raise CliParameterError("resume-policy must be strict, auto, weights-only, or none")
    selected = _select_device(device)
    settings = load_settings(config)
    inheritance = load_inheritance_package(
        inherit_from=inherit_from,
        current_settings=settings,
        policy=resume_policy,  # type: ignore[arg-type]
    )
    return run_training_ramp(
        config_path=config,
        device=str(selected),
        output_root=output_root,
        target_config_path=target_config,
        run_id=run_id,
        auto_launch_full=auto_launch_full,
        force_level=force_level,
        max_levels=max_levels,
        run_full_checks=run_full_checks,
        progress_ui=progress_ui,
        progress_language=progress_language,
        resume_policy=resume_policy,
        resume_stage_checkpoints=inheritance.stage_checkpoint_paths,
    )


def run_full_flow_job(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    device: str = "auto",
    mode: str = "execute",
    output_root: Path = DEFAULT_FULL_FLOW_ROOT,
    target_config: Path | None = None,
    run_id: str | None = None,
    auto_launch_full: bool = False,
    force_level: bool = False,
    max_levels: int | None = None,
    run_full_checks: bool = True,
    progress_ui: str = "auto",
    progress_language: str = "zh-CN",
    inherit_from: Path | str | None = None,
    resume_policy: str = "none",
    resume: bool = False,
    from_stage: str | None = None,
    until_stage: str | None = None,
    force_stages: tuple[str, ...] = (),
    skip_stages: tuple[str, ...] = (),
):
    if mode not in {"execute", "plan", "dry-run", "status"}:
        raise CliParameterError("mode must be execute, plan, dry-run, or status")
    selected_policy = "auto" if resume and resume_policy == "none" else resume_policy
    if selected_policy not in {"strict", "auto", "weights-only", "none"}:
        raise CliParameterError("resume-policy must be strict, auto, weights-only, or none")
    if progress_ui not in {"auto", "rich", "plain", "off"}:
        raise CliParameterError("progress-ui must be auto, rich, plain, or off")
    selected_inherit_from: Path | str | None = (
        "latest" if resume and inherit_from is None else inherit_from
    )
    if mode == "status":
        return load_full_flow_status(output_root, run_id=run_id)
    return run_full_flow(
        FullFlowConfig(
            config_path=config,
            device=device,
            mode=mode,  # type: ignore[arg-type]
            output_root=output_root,
            target_config_path=target_config,
            run_id=run_id,
            auto_launch_full=auto_launch_full,
            force_level=force_level,
            max_levels=max_levels,
            run_full_checks=run_full_checks,
            progress_ui=progress_ui,
            progress_language=progress_language,
            inherit_from=selected_inherit_from,
            resume_policy=selected_policy,
            resume_requested=resume,
            from_stage=from_stage,
            until_stage=until_stage,
            force_stages=force_stages,
            skip_stages=skip_stages,
        )
    )


def run_model_smoke(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    device: str = "cpu",
    backward: bool = True,
) -> dict[str, Any]:
    selected = _select_device(device)
    summary = _execute_model_smoke(
        config=config,
        device=selected,
        backward=backward,
    )
    output_dir = _run_dir("model_smoke")
    _write_summary_txt(output_dir, summary)
    summary["run_dir"] = output_dir
    return summary


def run_spatial_decode_smoke(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    split: DataSplit = "train",
    index: int = 0,
    device: str = "cpu",
    max_candidates: int = 16,
    score_threshold: float = 0.0,
    nms_radius_px: float = 32.0,
    slider_threshold: float = 0.5,
    max_slider_paths: int = 16,
    patch_limit: int | None = None,
) -> dict[str, Any]:
    settings = load_settings(config)
    selected = _select_device(device)
    dataset = build_dataset(settings, split=split)
    if index >= len(dataset):
        raise CliParameterError(
            f"index {index} is outside dataset length {len(dataset)}"
        )
    sample = dataset[index]
    result = run_spatial_frame_inference(
        settings,
        sample,
        device=selected,
        max_candidates=max_candidates,
        score_threshold=score_threshold,
        nms_radius_px=nms_radius_px,
        slider_threshold=slider_threshold,
        max_slider_paths=max_slider_paths,
        patch_limit=patch_limit,
    )
    output_dir = _run_dir("spatial_decode_smoke")
    candidate_rows = [
        spatial_candidate_to_dict(candidate) for candidate in result.candidates
    ]
    slider_rows = [slider_path_to_dict(path) for path in result.slider_paths]
    (output_dir / "candidates.json").write_text(
        json.dumps(candidate_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "slider_paths.json").write_text(
        json.dumps(slider_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = result.as_summary() | {
        "sample_key": sample["sample_key"],
        "frame_index": sample["frame_index"],
        "timestamp_ms": sample["timestamp_ms"],
        "top_candidate": candidate_rows[0] if candidate_rows else None,
        "top_slider_path": (
            _compact_slider_path(slider_rows[0]) if slider_rows else None
        ),
        "run_dir": output_dir,
    }
    _write_summary_txt(output_dir, summary)
    return summary


def run_candidate_cache_build(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    split: DataSplit = "train",
    device: str = "cpu",
    max_frames: int | None = None,
    patch_limit: int | None = None,
    max_candidates: int | None = None,
    score_threshold: float | None = None,
    nms_radius_px: float | None = None,
    slider_threshold: float | None = None,
    max_slider_paths: int | None = None,
    output: Path | None = None,
):
    settings = load_settings(config)
    selected = _select_device(device)
    output_dir = output or _run_dir(
        str(split), root=settings.candidate_cache.output_root
    )
    return generate_candidate_cache(
        settings,
        output_dir=output_dir,
        device=selected,
        split=split,
        max_frames=max_frames,
        patch_limit=patch_limit,
        max_candidates=max_candidates,
        score_threshold=score_threshold,
        nms_radius_px=nms_radius_px,
        slider_threshold=slider_threshold,
        max_slider_paths=max_slider_paths,
    )


def run_memory_profile(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    device: str = "cuda",
) -> dict[str, Any]:
    selected = _select_device(device)
    summary = _execute_model_smoke(
        config=config,
        device=selected,
        backward=True,
    )
    output_dir = _run_dir("memory_profile")
    _write_summary_txt(output_dir, summary)
    summary["run_dir"] = output_dir
    return summary


def visualize_patch_windows(
    *,
    input_image: Path,
    output: Path | None = None,
    config: Path = DEFAULT_TRAINING_CONFIG,
) -> Path:
    settings = load_settings(config)
    image = Image.open(input_image).convert("RGB")
    stream = PatchStream(
        patch_width=settings.tiling.patch_width,
        patch_height=settings.tiling.patch_height,
        overlap_x=settings.tiling.overlap_x,
        overlap_y=settings.tiling.overlap_y,
    )
    draw = ImageDraw.Draw(image)
    for meta in stream.metas(frame_width=image.width, frame_height=image.height):
        draw.rectangle(
            (meta.x0, meta.y0, meta.x1 - 1, meta.y1 - 1), outline="red", width=2
        )
    output_path = output or (_run_dir("visualize_patches") / "patches.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def visualize_fusion_context(
    *,
    input_image: Path,
    output: Path | None = None,
    config: Path = DEFAULT_TRAINING_CONFIG,
    device: str = "cpu",
) -> Path:
    settings = load_settings(config)
    selected = _select_device(device)
    enforce_runtime_memory_budget(
        device=selected,
        max_vram_gib=settings.memory.max_vram_gib,
        reserve_vram_gib=settings.memory.reserve_vram_gib,
        max_ram_gib=settings.memory.max_ram_gib,
        reserve_ram_gib=settings.memory.reserve_ram_gib,
    )
    runtime_state = configure_torch_runtime(
        device=selected,
        amp_dtype=settings.memory.amp_dtype,
        runtime=CudaRuntimeConfig(
            allow_tf32=settings.memory.allow_tf32,
            cudnn_benchmark=settings.memory.cudnn_benchmark,
            matmul_float32_precision=settings.memory.matmul_float32_precision,
            channels_last=settings.memory.channels_last,
        ),
    )
    frame = append_color_cues(
        _load_image_tensor(input_image),
        mode=settings.input.color_cues,
    )
    stream = PatchStream(
        patch_width=settings.tiling.patch_width,
        patch_height=settings.tiling.patch_height,
        overlap_x=settings.tiling.overlap_x,
        overlap_y=settings.tiling.overlap_y,
    )
    patch, meta = next(stream.iter_patches(frame))
    modules = _build_model_stack(settings)
    for name, module in tuple(modules.items()):
        moved = module_to_device(
            module,
            selected,
            channels_last=runtime_state.channels_last,
        )
        moved.eval()
        modules[name] = moved
    frame_device = tensor_to_device(
        frame.unsqueeze(0),
        selected,
        channels_last=runtime_state.channels_last,
        non_blocking=False,
    )
    patch_device = tensor_to_device(
        patch.unsqueeze(0),
        selected,
        channels_last=runtime_state.channels_last,
        non_blocking=False,
    )
    with torch.no_grad():
        global_features = modules["global"](frame_device)
        local_features = modules["local"](patch_device)
        fused = modules["fusion"](
            local_features=local_features,
            global_features=global_features.dense,
            patch_meta=meta,
        )
        heatmap = fused.global_context.mean(dim=1, keepdim=True)
        heatmap = torch.nn.functional.interpolate(
            heatmap,
            size=(settings.tiling.patch_height, settings.tiling.patch_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    heatmap = heatmap.detach().cpu()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()).clamp_min(
        1e-6
    )
    heatmap_image = Image.fromarray((heatmap * 255).byte().numpy(), mode="L")
    output_path = output or (_run_dir("visualize_fusion") / "fusion_context.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_image.save(output_path)
    return output_path


def run_spatial_training_job(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    split: DataSplit = "train",
    device: str = "auto",
    max_steps: int = 1,
    learning_rate: float = 1e-4,
    patch_limit: int | None = None,
):
    settings = load_settings(config)
    selected = _select_device(device)
    output_dir = _run_dir("train_spatial")
    return run_spatial_training(
        settings,
        device=selected,
        run_dir=output_dir,
        split=split,
        max_steps=max_steps,
        learning_rate=learning_rate,
        patch_limit=patch_limit,
    )


def spatial_training_oom_guidance(config: Path) -> str:
    settings = load_settings(config)
    return format_oom_guidance(
        patch_size=(
            settings.tiling.patch_width,
            settings.tiling.patch_height,
        ),
        global_size=(
            settings.global_encoder.input_width,
            settings.global_encoder.input_height,
        ),
        batch_size=1,
        amp_dtype=settings.memory.amp_dtype,
        config_path=str(config),
    )


def run_temporal_training_job(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    cache: Path,
    device: str = "auto",
    max_steps: int = 1,
    learning_rate: float = 1e-4,
    sequence_length: int | None = None,
    candidate_slots: int | None = None,
):
    settings = load_settings(config)
    selected = _select_device(device)
    output_dir = _run_dir("train_temporal")
    return run_temporal_training(
        settings,
        cache_dir=cache,
        device=selected,
        run_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        sequence_length=sequence_length,
        candidate_slots=candidate_slots,
    )


def run_decision_job(
    *,
    config: Path = DEFAULT_TRAINING_CONFIG,
    cache: Path,
    checkpoint: Path,
    output: Path | None = None,
    device: str = "auto",
):
    settings = load_settings(config)
    selected = _select_device(device)
    output_dir = output or _run_dir("decision")
    return run_temporal_decision(
        settings,
        cache_dir=cache,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        device=selected,
    )


def run_label_visualization(
    *,
    segment_index: int = 0,
    object_index: int = 0,
    output: Path | None = None,
    show: bool = False,
    config: Path | None = None,
):
    return visualize_click_label(
        load_settings(config),
        segment_index=segment_index,
        object_index=object_index,
        output_path=output,
        show_window=show,
    )


def run_gallery_export(
    *,
    results: Path,
    output_root: Path | None = None,
    samples_per_group: int | None = None,
    config: Path | None = None,
):
    return save_annotation_gallery(
        load_settings(config),
        load_batch_gallery_request(results),
        output_root=output_root,
        samples_per_group=samples_per_group,
    )


def _raise_cli_parameter(error: CliParameterError) -> NoReturn:
    raise typer.BadParameter(str(error)) from error


@app.command("data-check")
def data_check(
    config: Path | None = typer.Option(None, "--config"),
    split: DataSplit = typer.Option("all", "--split"),
) -> None:
    settings, report = inspect_training_data(config=config, split=split)
    _render_report(report)
    if settings.data_input.strict and not report.ok:
        raise typer.Exit(1)


@app.command("env-check")
def env_check(
    strict: bool = typer.Option(
        False,
        "--strict/--no-strict",
        help="Exit non-zero when required runtime dependencies are missing.",
    ),
    require_cuda: bool = typer.Option(
        False,
        "--require-cuda/--no-require-cuda",
        help="Treat CUDA unavailability as a failure in strict mode.",
    ),
) -> None:
    report = collect_training_environment()
    _render_env_report(report)
    if strict and not report.ready(require_cuda=require_cuda):
        missing = ", ".join(report.missing_required_packages) or "none"
        console.print(f"[red]environment check failed; missing: {missing}[/red]")
        if require_cuda and not report.torch.cuda_available:
            console.print("[red]CUDA is not available inside the container[/red]")
        raise typer.Exit(1)


@app.command("data-preview")
def data_preview(
    index: int = typer.Option(0, "--index", min=0),
    split: DataSplit = typer.Option("train", "--split"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    try:
        summary = preview_training_sample(
            index=index,
            split=split,
            config=config,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    console.print(summary)


@app.command("run")
def run(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    split: DataSplit = typer.Option("train", "--split"),
    device: str = typer.Option(
        "auto",
        "--device",
        help="cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.",
    ),
    spatial_max_steps: int = typer.Option(1, "--spatial-max-steps", min=1),
    temporal_max_steps: int = typer.Option(1, "--temporal-max-steps", min=1),
    spatial_learning_rate: float = typer.Option(
        1e-4,
        "--spatial-lr",
        min=1e-8,
    ),
    temporal_learning_rate: float = typer.Option(
        1e-4,
        "--temporal-lr",
        min=1e-8,
    ),
    patch_limit: int = typer.Option(
        1,
        "--patch-limit",
        min=0,
        help="0 means process all patches in each frame.",
    ),
    cache_max_frames: int = typer.Option(
        1,
        "--cache-max-frames",
        min=0,
        help="0 means no frame limit for candidate cache generation.",
    ),
    sequence_length: int | None = typer.Option(None, "--sequence-length", min=1),
    candidate_slots: int | None = typer.Option(None, "--candidate-slots", min=1),
    parameter_group_id: str = typer.Option("pg-0001", "--parameter-group-id"),
    render_gallery: bool = typer.Option(
        True,
        "--render-gallery/--no-render-gallery",
        help="Render the best parameter group gallery after the training round.",
    ),
    gallery_output_root: Path | None = typer.Option(None, "--gallery-output-root"),
    gallery_samples_per_group: int | None = typer.Option(
        None,
        "--gallery-samples-per-group",
        min=1,
    ),
    progress_ui: str = typer.Option("auto", "--progress-ui"),
    progress_language: str = typer.Option("zh-CN", "--progress-language"),
    inherit_from: str | None = typer.Option(None, "--inherit-from"),
    resume_policy: str = typer.Option("none", "--resume-policy"),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    try:
        selected_inherit_from: str | None = (
            "latest" if resume and inherit_from is None else inherit_from
        )
        selected_policy = "auto" if resume and resume_policy == "none" else resume_policy
        result = run_training(
            config=config,
            split=split,
            device=device,
            spatial_max_steps=spatial_max_steps,
            temporal_max_steps=temporal_max_steps,
            spatial_learning_rate=spatial_learning_rate,
            temporal_learning_rate=temporal_learning_rate,
            patch_limit=patch_limit,
            cache_max_frames=cache_max_frames,
            sequence_length=sequence_length,
            candidate_slots=candidate_slots,
            parameter_group_id=parameter_group_id,
            render_gallery=render_gallery,
            gallery_output_root=gallery_output_root,
            gallery_samples_per_group=gallery_samples_per_group,
            progress_ui=progress_ui,
            progress_language=progress_language,
            inherit_from=selected_inherit_from,
            resume_policy=selected_policy,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    _render_dict_table("Full training pipeline", result.as_summary())
    _render_parameter_group_score(result.evaluation)


@app.command("run-job")
def run_job(
    job: Path = typer.Option(
        ...,
        "--job",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    device: str = typer.Option("auto", "--device"),
    execute: bool = typer.Option(True, "--execute/--dry-run"),
) -> None:
    try:
        result = run_training_job_spec(
            job=job,
            config=config,
            device=device,
            execute=execute,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    _render_dict_table("Training job", result)


@app.command("full-flow")
def full_flow(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    device: str = typer.Option(
        "auto",
        "--device",
        help="cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.",
    ),
    mode: str = typer.Option(
        "execute",
        "--mode",
        help="execute, plan, dry-run, or status.",
    ),
    output_root: Path = typer.Option(DEFAULT_FULL_FLOW_ROOT, "--output-root"),
    target_config: Path | None = typer.Option(None, "--target-config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    auto_launch_full: bool = typer.Option(
        False,
        "--auto-launch-full/--no-auto-launch-full",
        help="Launch finite full training after ramp gates pass.",
    ),
    force_level: bool = typer.Option(
        False,
        "--force-level/--resume-passed-levels",
    ),
    max_levels: int | None = typer.Option(None, "--max-levels", min=1),
    run_full_checks: bool = typer.Option(
        True,
        "--run-full-checks/--skip-full-checks",
    ),
    progress_ui: str = typer.Option("auto", "--progress-ui"),
    progress_language: str = typer.Option("zh-CN", "--progress-language"),
    inherit_from: str | None = typer.Option(None, "--inherit-from"),
    resume_policy: str = typer.Option("none", "--resume-policy"),
    resume: bool = typer.Option(False, "--resume"),
    from_stage: str | None = typer.Option(None, "--from-stage"),
    until_stage: str | None = typer.Option(None, "--until-stage"),
    force_stage: list[str] | None = typer.Option(None, "--force-stage"),
    skip_stage: list[str] | None = typer.Option(None, "--skip-stage"),
) -> None:
    try:
        result = run_full_flow_job(
            config=config,
            device=device,
            mode=mode,
            output_root=output_root,
            target_config=target_config,
            run_id=run_id,
            auto_launch_full=auto_launch_full,
            force_level=force_level,
            max_levels=max_levels,
            run_full_checks=run_full_checks,
            progress_ui=progress_ui,
            progress_language=progress_language,
            inherit_from=inherit_from,
            resume_policy=resume_policy,
            resume=resume,
            from_stage=from_stage,
            until_stage=until_stage,
            force_stages=tuple(force_stage or ()),
            skip_stages=tuple(skip_stage or ()),
        )
    except (CliParameterError, ValueError) as error:
        _raise_cli_parameter(CliParameterError(str(error)))
    _render_dict_table("Full flow", result.as_dict())


@app.command("full-flow-status")
def full_flow_status(
    output_root: Path = typer.Option(DEFAULT_FULL_FLOW_ROOT, "--output-root"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    try:
        result = load_full_flow_status(output_root, run_id=run_id)
    except FileNotFoundError as error:
        _raise_cli_parameter(CliParameterError(str(error)))
    _render_dict_table("Full flow status", result.as_dict())


@app.command("ramp-to-full")
def ramp_to_full(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    device: str = typer.Option(
        "auto",
        "--device",
        help="cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.",
    ),
    output_root: Path = typer.Option(
        Path("artifacts") / "training_ramp",
        "--output-root",
    ),
    target_config: Path | None = typer.Option(None, "--target-config"),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Resume or extend an existing ramp output run id.",
    ),
    auto_launch_full: bool = typer.Option(
        False,
        "--auto-launch-full/--no-auto-launch-full",
        help="Launch the finite full training run after all ramp gates pass.",
    ),
    force_level: bool = typer.Option(
        False,
        "--force-level/--resume-passed-levels",
        help="Re-run levels even when their level_state.json is already passed.",
    ),
    max_levels: int | None = typer.Option(
        None,
        "--max-levels",
        min=1,
        help="Limit levels for controlled validation; omit for target ramp.",
    ),
    run_full_checks: bool = typer.Option(
        True,
        "--run-full-checks/--skip-full-checks",
        help="Run full pytest checks during preflight.",
    ),
    progress_ui: str = typer.Option("auto", "--progress-ui"),
    progress_language: str = typer.Option("zh-CN", "--progress-language"),
    inherit_from: str | None = typer.Option(None, "--inherit-from"),
    resume_policy: str = typer.Option("none", "--resume-policy"),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    try:
        selected_inherit_from: str | None = (
            "latest" if resume and inherit_from is None else inherit_from
        )
        selected_policy = "auto" if resume and resume_policy == "none" else resume_policy
        result = run_training_ramp_job(
            config=config,
            device=device,
            output_root=output_root,
            target_config=target_config,
            run_id=run_id,
            auto_launch_full=auto_launch_full,
            force_level=force_level,
            max_levels=max_levels,
            run_full_checks=run_full_checks,
            progress_ui=progress_ui,
            progress_language=progress_language,
            inherit_from=selected_inherit_from,
            resume_policy=selected_policy,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    _render_dict_table("Training ramp", result.as_dict())


@app.command("model-smoke")
def model_smoke(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="cpu, cuda, or auto. CPU is the default smoke path.",
    ),
    backward: bool = typer.Option(
        True,
        "--backward/--no-backward",
        help="Run backward and optimizer step in addition to forward.",
    ),
) -> None:
    try:
        summary = run_model_smoke(
            config=config,
            device=device,
            backward=backward,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    _render_dict_table("Model smoke", summary)


@app.command("spatial-decode-smoke")
def spatial_decode_smoke(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    split: DataSplit = typer.Option("train", "--split"),
    index: int = typer.Option(0, "--index", min=0),
    device: str = typer.Option("cpu", "--device"),
    max_candidates: int = typer.Option(16, "--max-candidates", min=1),
    score_threshold: float = typer.Option(0.0, "--score-threshold", min=0.0),
    nms_radius_px: float = typer.Option(32.0, "--nms-radius-px", min=0.0),
    slider_threshold: float = typer.Option(0.5, "--slider-threshold", min=0.0, max=1.0),
    max_slider_paths: int = typer.Option(16, "--max-slider-paths", min=1),
    patch_limit: int | None = typer.Option(None, "--patch-limit", min=1),
) -> None:
    try:
        summary = run_spatial_decode_smoke(
            config=config,
            split=split,
            index=index,
            device=device,
            max_candidates=max_candidates,
            score_threshold=score_threshold,
            nms_radius_px=nms_radius_px,
            slider_threshold=slider_threshold,
            max_slider_paths=max_slider_paths,
            patch_limit=patch_limit,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    _render_dict_table("Spatial decode smoke", summary)


@app.command("build-candidate-cache")
def build_candidate_cache(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    split: DataSplit = typer.Option("train", "--split"),
    device: str = typer.Option("cpu", "--device"),
    max_frames: int | None = typer.Option(None, "--max-frames", min=1),
    patch_limit: int | None = typer.Option(None, "--patch-limit", min=1),
    max_candidates: int | None = typer.Option(None, "--max-candidates", min=1),
    score_threshold: float | None = typer.Option(
        None,
        "--score-threshold",
        min=0.0,
        max=1.0,
    ),
    nms_radius_px: float | None = typer.Option(None, "--nms-radius-px", min=0.0),
    slider_threshold: float | None = typer.Option(
        None,
        "--slider-threshold",
        min=0.0,
        max=1.0,
    ),
    max_slider_paths: int | None = typer.Option(None, "--max-slider-paths", min=1),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    try:
        result = run_candidate_cache_build(
            config=config,
            split=split,
            device=device,
            max_frames=max_frames,
            patch_limit=patch_limit,
            max_candidates=max_candidates,
            score_threshold=score_threshold,
            nms_radius_px=nms_radius_px,
            slider_threshold=slider_threshold,
            max_slider_paths=max_slider_paths,
            output=output,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    _render_dict_table("Candidate cache", result.as_dict())


@app.command("memory-profile")
def memory_profile(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="cuda, cpu, or auto. CUDA is the default for memory profiling.",
    ),
) -> None:
    try:
        selected = _select_device(device)
        summary = run_memory_profile(config=config, device=device)
    except CliParameterError as error:
        _raise_cli_parameter(error)
    if selected.type != "cuda":
        console.print(
            "[yellow]CPU profile completed; CUDA memory is unavailable.[/yellow]"
        )
    _render_dict_table("Memory profile", summary)


@app.command("visualize-patches")
def visualize_patches(
    input_image: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path | None = typer.Option(None, "--output"),
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
) -> None:
    output_path = visualize_patch_windows(
        input_image=input_image,
        output=output,
        config=config,
    )
    console.print(f"[green]saved[/green]: {output_path}")


@app.command("visualize-fusion")
def visualize_fusion(
    input_image: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path | None = typer.Option(None, "--output"),
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    try:
        output_path = visualize_fusion_context(
            input_image=input_image,
            output=output,
            config=config,
            device=device,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    console.print(f"[green]saved[/green]: {output_path}")


@app.command("train-spatial")
def train_spatial(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    split: DataSplit = typer.Option("train", "--split"),
    device: str = typer.Option(
        "auto",
        "--device",
        help="cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.",
    ),
    max_steps: int = typer.Option(1, "--max-steps", min=1),
    learning_rate: float = typer.Option(1e-4, "--lr", min=1e-8),
    patch_limit: int | None = typer.Option(None, "--patch-limit", min=1),
) -> None:
    try:
        result = run_spatial_training_job(
            config=config,
            device=device,
            split=split,
            max_steps=max_steps,
            learning_rate=learning_rate,
            patch_limit=patch_limit,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            console.print("[red]" + spatial_training_oom_guidance(config) + "[/red]")
        raise
    _render_dict_table("Spatial training", result.as_dict())


@app.command("train-temporal")
def train_temporal(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    cache: Path = typer.Option(
        ...,
        "--cache",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Candidate cache directory containing manifest.json and frames.jsonl.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.",
    ),
    max_steps: int = typer.Option(1, "--max-steps", min=1),
    learning_rate: float = typer.Option(1e-4, "--lr", min=1e-8),
    sequence_length: int | None = typer.Option(None, "--sequence-length", min=1),
    candidate_slots: int | None = typer.Option(None, "--candidate-slots", min=1),
) -> None:
    try:
        result = run_temporal_training_job(
            config=config,
            cache=cache,
            device=device,
            max_steps=max_steps,
            learning_rate=learning_rate,
            sequence_length=sequence_length,
            candidate_slots=candidate_slots,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    _render_dict_table("Temporal training", result.as_dict())


@app.command("run-decision")
def run_decision(
    config: Path = typer.Option(DEFAULT_TRAINING_CONFIG, "--config"),
    cache: Path = typer.Option(
        ...,
        "--cache",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Candidate cache directory containing manifest.json and frames.jsonl.",
    ),
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Temporal checkpoint produced by train-temporal.",
    ),
    output: Path | None = typer.Option(None, "--output"),
    device: str = typer.Option(
        "auto",
        "--device",
        help="cpu, cuda, or auto. Use cuda through host-exec for real GPU runs.",
    ),
) -> None:
    try:
        result = run_decision_job(
            config=config,
            cache=cache,
            checkpoint=checkpoint,
            output=output,
            device=device,
        )
    except CliParameterError as error:
        _raise_cli_parameter(error)
    _render_dict_table("Temporal decision", result.as_dict())


@app.command("visualize-label")
def visualize_label(
    segment_index: int = typer.Option(0, "--segment-index", min=0),
    object_index: int = typer.Option(0, "--object-index", min=0),
    output: Path | None = typer.Option(None, "--output"),
    show: bool = typer.Option(False, "--show/--no-show"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    result = run_label_visualization(
        config=config,
        segment_index=segment_index,
        object_index=object_index,
        output=output,
        show=show,
    )
    if result.output_path is not None:
        console.print(f"[green]{result.status}[/green]: {result.output_path}")
    else:
        console.print(result.status)
    if result.warning:
        console.print(f"[yellow]warning:[/yellow] {result.warning}")


@app.command("save-annotation-gallery")
def save_gallery(
    results: Path = typer.Option(
        ...,
        "--results",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_root: Path | None = typer.Option(None, "--output-root"),
    samples_per_group: int | None = typer.Option(
        None,
        "--samples-per-group",
        min=1,
    ),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    result = run_gallery_export(
        config=config,
        results=results,
        output_root=output_root,
        samples_per_group=samples_per_group,
    )
    if result.output_dir is not None:
        console.print(
            f"[green]{result.status}[/green]: {result.output_dir} "
            f"({result.saved_frame_count} frames, "
            f"trial={result.selected_trial_id})"
        )
    else:
        console.print(result.status)
    if result.warning:
        console.print(f"[yellow]warning:[/yellow] {result.warning}")
    if not result.succeeded:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
