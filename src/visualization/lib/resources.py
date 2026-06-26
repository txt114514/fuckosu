from __future__ import annotations

import shutil
import subprocess

import psutil
import torch

from visualization.lib.models import ResourceState


def collect_resource_state() -> ResourceState:
    disk = shutil.disk_usage(".")
    process = psutil.Process()
    state = ResourceState(
        cpu_percent=psutil.cpu_percent(interval=None),
        system_memory_gb=psutil.virtual_memory().used / 1024**3,
        process_memory_gb=process.memory_info().rss / 1024**3,
        disk_free_gb=disk.free / 1024**3,
    )
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        state.gpu_name = props.name
        state.gpu_total_gb = props.total_memory / 1024**3
        state.gpu_allocated_gb = torch.cuda.memory_allocated(index) / 1024**3
        state.gpu_reserved_gb = torch.cuda.memory_reserved(index) / 1024**3
        state.gpu_peak_allocated_gb = torch.cuda.max_memory_allocated(index) / 1024**3
        state.gpu_peak_reserved_gb = torch.cuda.max_memory_reserved(index) / 1024**3
        _attach_nvidia_smi(state)
    return state


def _attach_nvidia_smi(state: ResourceState) -> None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except Exception:
        return
    if completed.returncode != 0 or not completed.stdout.strip():
        return
    first = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) >= 3:
        try:
            state.gpu_utilization = float(parts[0])
            state.gpu_temperature_c = float(parts[1])
            state.gpu_power_w = float(parts[2])
        except ValueError:
            return
