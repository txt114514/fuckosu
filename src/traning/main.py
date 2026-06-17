from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from PIL import Image, ImageDraw
import typer
from rich.console import Console
from rich.table import Table

from traning.Lib.data import build_patch_windows
from traning.Lib.data import PatchStream, append_color_cues
from traning.Lib.models import build_model_stack
from traning.Lib.runtime import (
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
from traning.core.candidate_cache import generate_candidate_cache
from traning.core.dataset_import import build_dataset, inspect_data_input
from traning.core.pipeline import run_pipeline
from traning.core.spatial_training import (
    run_spatial_frame_inference,
    run_spatial_training,
    slider_path_to_dict,
    spatial_candidate_to_dict,
)
from traning.core.visualization import (
    save_annotation_gallery,
    visualize_click_label,
)
from traning.state import load_batch_gallery_request
from environment import collect_environment_report


app = typer.Typer(help="osu! video model training commands.")
console = Console()


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
        raise typer.BadParameter("CUDA is not available inside the container")
    if selected not in {"cpu", "cuda"}:
        raise typer.BadParameter("device must be cpu, cuda, or auto")
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


@app.command("data-check")
def data_check(
    config: Path | None = typer.Option(None, "--config"),
    split: DataSplit = typer.Option("all", "--split"),
) -> None:
    settings = load_settings(config)
    report = inspect_data_input(settings, split=split)
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
    report = collect_environment_report()
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
    settings = load_settings(config)
    dataset = build_dataset(settings, split=split)
    if index >= len(dataset):
        raise typer.BadParameter(
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
    console.print(
        {
            "sample_key": sample["sample_key"],
            "frame_index": sample["frame_index"],
            "timestamp_ms": sample["timestamp_ms"],
            "image_shape": tuple(sample["image"].shape),
            "visible_objects": len(sample["visible_hit_objects"]),
            "patch_count": len(windows),
            "first_patch": windows[0],
        }
    )


@app.command("run")
def run(
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    results = run_pipeline(load_settings(config))
    _render_report(results["data_input"])


@app.command("model-smoke")
def model_smoke(
    config: Path = typer.Option(Path("configs/model_small_vram.yaml"), "--config"),
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
    selected = _select_device(device)
    summary = _execute_model_smoke(
        config=config,
        device=selected,
        backward=backward,
    )
    output_dir = _run_dir("model_smoke")
    (output_dir / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    summary["run_dir"] = output_dir
    _render_dict_table("Model smoke", summary)


@app.command("spatial-decode-smoke")
def spatial_decode_smoke(
    config: Path = typer.Option(Path("configs/model_small_vram.yaml"), "--config"),
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
    settings = load_settings(config)
    selected = _select_device(device)
    dataset = build_dataset(settings, split=split)
    if index >= len(dataset):
        raise typer.BadParameter(
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
    (output_dir / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    _render_dict_table("Spatial decode smoke", summary)


@app.command("build-candidate-cache")
def build_candidate_cache(
    config: Path = typer.Option(Path("configs/model_small_vram.yaml"), "--config"),
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
    settings = load_settings(config)
    selected = _select_device(device)
    output_dir = output or _run_dir(
        str(split), root=settings.candidate_cache.output_root
    )
    result = generate_candidate_cache(
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
    _render_dict_table("Candidate cache", result.as_dict())


@app.command("memory-profile")
def memory_profile(
    config: Path = typer.Option(Path("configs/model_small_vram.yaml"), "--config"),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="cuda, cpu, or auto. CUDA is the default for memory profiling.",
    ),
) -> None:
    selected = _select_device(device)
    summary = _execute_model_smoke(
        config=config,
        device=selected,
        backward=True,
    )
    output_dir = _run_dir("memory_profile")
    (output_dir / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    if selected.type != "cuda":
        console.print(
            "[yellow]CPU profile completed; CUDA memory is unavailable.[/yellow]"
        )
    summary["run_dir"] = output_dir
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
    config: Path = typer.Option(Path("configs/model_small_vram.yaml"), "--config"),
) -> None:
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
    config: Path = typer.Option(Path("configs/model_small_vram.yaml"), "--config"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
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
    console.print(f"[green]saved[/green]: {output_path}")


def _training_placeholder(stage: str, config: Path) -> None:
    settings = load_settings(config)
    output_dir = _run_dir(stage)
    (output_dir / "summary.txt").write_text(
        "\n".join(
            (
                f"stage: {stage}",
                f"config: {config}",
                f"device: {settings.runtime.device}",
                "status: training loop is not implemented yet",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(
        f"[yellow]{stage} training entry is registered; "
        f"full optimizer loop is still pending.[/yellow]"
    )
    console.print(f"run_dir: {output_dir}")
    raise typer.Exit(1)


@app.command("train-spatial")
def train_spatial(
    config: Path = typer.Option(Path("configs/model_small_vram.yaml"), "--config"),
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
    settings = load_settings(config)
    selected = _select_device(device)
    output_dir = _run_dir("train_spatial")
    try:
        result = run_spatial_training(
            settings,
            device=selected,
            run_dir=output_dir,
            split=split,
            max_steps=max_steps,
            learning_rate=learning_rate,
            patch_limit=patch_limit,
        )
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
                    config_path=str(config),
                )
                + "[/red]"
            )
        raise
    _render_dict_table("Spatial training", result.as_dict())


@app.command("train-fusion")
def train_fusion(
    config: Path = typer.Option(Path("configs/model_small_vram.yaml"), "--config"),
) -> None:
    _training_placeholder("train_fusion", config)


@app.command("train-temporal")
def train_temporal(
    config: Path = typer.Option(Path("configs/model_small_vram.yaml"), "--config"),
) -> None:
    _training_placeholder("train_temporal", config)


@app.command("visualize-label")
def visualize_label(
    segment_index: int = typer.Option(0, "--segment-index", min=0),
    object_index: int = typer.Option(0, "--object-index", min=0),
    output: Path | None = typer.Option(None, "--output"),
    show: bool = typer.Option(False, "--show/--no-show"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    result = visualize_click_label(
        load_settings(config),
        segment_index=segment_index,
        object_index=object_index,
        output_path=output,
        show_window=show,
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
    result = save_annotation_gallery(
        load_settings(config),
        load_batch_gallery_request(results),
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
