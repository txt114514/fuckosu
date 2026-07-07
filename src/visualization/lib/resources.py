from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Callable

import psutil
import torch

from visualization.lib.models import ResourceState


_BYTES_PER_GB = 1024**3
_MIB_PER_GB = 1024
_GPU_HISTORY_WINDOW = 60
_GPU_UTILIZATION_HISTORY: dict[int, deque[float]] = defaultdict(
    lambda: deque(maxlen=_GPU_HISTORY_WINDOW)
)


@dataclass(frozen=True)
class _GpuMonitorSample:
    index: int
    source: str
    name: str | None = None
    utilization: float | None = None
    memory_used_gb: float | None = None
    memory_total_gb: float | None = None
    memory_utilization: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    warning: str | None = None


@dataclass(frozen=True)
class _MonitorProbe:
    sample: _GpuMonitorSample | None = None
    error: str | None = None


def collect_resource_state(device_index: int | None = None) -> ResourceState:
    disk = shutil.disk_usage(".")
    process = psutil.Process()
    state = ResourceState(
        cpu_percent=psutil.cpu_percent(interval=None),
        system_memory_gb=psutil.virtual_memory().used / _BYTES_PER_GB,
        process_memory_gb=process.memory_info().rss / _BYTES_PER_GB,
        disk_free_gb=disk.free / _BYTES_PER_GB,
    )
    monitor_index = _attach_torch_cuda_state(state, device_index)
    torch_cuda_error = state.gpu_monitor_error
    if monitor_index is None:
        monitor_index = device_index if device_index is not None else 0
        state.gpu_index = monitor_index

    errors: list[str] = []
    for probe in _monitor_probes(monitor_index):
        if probe.sample is not None:
            _apply_monitor_sample(state, probe.sample)
            state.gpu_monitor_error = probe.sample.warning
            return state
        if probe.error:
            errors.append(probe.error)

    if torch_cuda_error:
        errors.insert(0, torch_cuda_error)
    state.gpu_monitor_error = _join_monitor_errors(errors)
    return state


def _attach_torch_cuda_state(
    state: ResourceState,
    device_index: int | None,
) -> int | None:
    try:
        if not torch.cuda.is_available():
            return None
        index = device_index if device_index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        state.gpu_index = index
        state.gpu_name = props.name
        state.gpu_total_gb = props.total_memory / _BYTES_PER_GB
        state.gpu_allocated_gb = torch.cuda.memory_allocated(index) / _BYTES_PER_GB
        state.gpu_reserved_gb = torch.cuda.memory_reserved(index) / _BYTES_PER_GB
        state.gpu_peak_allocated_gb = (
            torch.cuda.max_memory_allocated(index) / _BYTES_PER_GB
        )
        state.gpu_peak_reserved_gb = torch.cuda.max_memory_reserved(index) / _BYTES_PER_GB
        return index
    except Exception as error:
        state.gpu_monitor_error = f"PyTorch CUDA 状态读取失败：{error}"
        return device_index


def _monitor_probes(index: int) -> tuple[_MonitorProbe, ...]:
    probes: list[_MonitorProbe] = [_collect_pynvml(index), _collect_nvidia_smi(index)]
    bridge = _collect_host_exec_nvidia_smi(index)
    if bridge.error != "未配置 GPU 监控桥接" or bridge.sample is not None:
        probes.append(bridge)
    return tuple(probes)


def _collect_pynvml(index: int) -> _MonitorProbe:
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception as error:
        return _MonitorProbe(error=f"NVML Python 接口不可用：{error}")
    try:
        pynvml.nvmlInit()
        count = int(pynvml.nvmlDeviceGetCount())
        if count <= 0:
            return _MonitorProbe(error="NVML 未发现 GPU")
        selected = min(max(index, 0), count - 1)
        handle = pynvml.nvmlDeviceGetHandleByIndex(selected)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        temperature = _nvml_value(
            lambda: pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU,
            )
        )
        power = _nvml_value(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))
        sample = _GpuMonitorSample(
            index=selected,
            source="NVML",
            name=str(name) if name else None,
            utilization=float(util.gpu),
            memory_used_gb=float(memory.used) / _BYTES_PER_GB,
            memory_total_gb=float(memory.total) / _BYTES_PER_GB,
            memory_utilization=float(util.memory),
            temperature_c=float(temperature) if temperature is not None else None,
            power_w=(float(power) / 1000.0) if power is not None else None,
        )
        return _MonitorProbe(sample=sample)
    except Exception as error:
        return _MonitorProbe(error=f"NVML 读取失败：{error}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _collect_nvidia_smi(index: int) -> _MonitorProbe:
    if shutil.which("nvidia-smi") is None:
        return _MonitorProbe(error="未找到 nvidia-smi")
    return _run_nvidia_smi(
        index,
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name",
            "--format=csv,noheader,nounits",
        ],
        source="nvidia-smi",
        timeout=3,
    )


def _collect_host_exec_nvidia_smi(index: int) -> _MonitorProbe:
    if os.environ.get("VISUALIZATION_GPU_MONITOR_BRIDGE", "auto") == "off":
        return _MonitorProbe(error="未配置 GPU 监控桥接")
    if shutil.which("host-exec") is None:
        return _MonitorProbe(error="未找到 host-exec GPU 监控桥接")
    container = os.environ.get("VISUALIZATION_GPU_MONITOR_CONTAINER", "osu_ai_dev")
    user = os.environ.get("VISUALIZATION_GPU_MONITOR_USER", "dev")
    cwd = Path.cwd()
    query = (
        "nvidia-smi "
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw,name "
        "--format=csv,noheader,nounits"
    )
    command = [
        "host-exec",
        "docker",
        "exec",
        "-u",
        user,
        container,
        "bash",
        "-lc",
        f"cd {shlex.quote(str(cwd))} && {query}",
    ]
    return _run_nvidia_smi(
        index,
        command,
        source="host-exec nvidia-smi",
        timeout=8,
    )


def _run_nvidia_smi(
    index: int,
    command: list[str],
    *,
    source: str,
    timeout: float,
) -> _MonitorProbe:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except Exception as error:
        return _MonitorProbe(error=f"{source} 执行失败：{error}")
    if completed.returncode != 0:
        detail = _monitor_error_detail((completed.stderr or completed.stdout or "").strip())
        suffix = f"：{detail}" if detail else ""
        return _MonitorProbe(error=f"{source} 返回失败码 {completed.returncode}{suffix}")
    return _parse_nvidia_smi_output(completed.stdout, index=index, source=source)


def _parse_nvidia_smi_output(output: str, *, index: int, source: str) -> _MonitorProbe:
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if not rows:
        return _MonitorProbe(error=f"{source} 未返回 GPU 监控数据")
    parsed = [_parse_nvidia_smi_row(row, source=source) for row in rows]
    samples = [sample for sample in parsed if sample is not None]
    if not samples:
        return _MonitorProbe(error=f"{source} 返回的数据无法解析")
    selected = next((sample for sample in samples if sample.index == index), samples[0])
    if selected.utilization is None:
        selected = _GpuMonitorSample(
            index=selected.index,
            source=selected.source,
            name=selected.name,
            utilization=selected.utilization,
            memory_used_gb=selected.memory_used_gb,
            memory_total_gb=selected.memory_total_gb,
            memory_utilization=selected.memory_utilization,
            temperature_c=selected.temperature_c,
            power_w=selected.power_w,
            warning=f"{source} 已连接，但 GPU 利用率字段不可用",
        )
    return _MonitorProbe(sample=selected)


def _parse_nvidia_smi_row(row: str, *, source: str) -> _GpuMonitorSample | None:
    parts = [part.strip() for part in row.split(",", maxsplit=6)]
    if len(parts) < 7:
        return None
    index = int(_parse_float(parts[0]) or 0)
    utilization = _parse_float(parts[1])
    memory_used_mib = _parse_float(parts[2])
    memory_total_mib = _parse_float(parts[3])
    temperature = _parse_float(parts[4])
    power = _parse_float(parts[5])
    memory_used_gb = (
        memory_used_mib / _MIB_PER_GB if memory_used_mib is not None else None
    )
    memory_total_gb = (
        memory_total_mib / _MIB_PER_GB if memory_total_mib is not None else None
    )
    memory_utilization = None
    if (
        memory_used_gb is not None
        and memory_total_gb is not None
        and memory_total_gb > 0
    ):
        memory_utilization = memory_used_gb / memory_total_gb * 100.0
    return _GpuMonitorSample(
        index=index,
        source=source,
        name=parts[6] or None,
        utilization=utilization,
        memory_used_gb=memory_used_gb,
        memory_total_gb=memory_total_gb,
        memory_utilization=memory_utilization,
        temperature_c=temperature,
        power_w=power,
    )


def _apply_monitor_sample(state: ResourceState, sample: _GpuMonitorSample) -> None:
    state.gpu_index = sample.index
    state.gpu_monitor_source = sample.source
    if sample.name:
        state.gpu_name = sample.name
    if sample.memory_total_gb is not None:
        state.gpu_total_gb = sample.memory_total_gb
    state.gpu_memory_used_gb = sample.memory_used_gb
    state.gpu_memory_utilization = sample.memory_utilization
    state.gpu_utilization = sample.utilization
    state.gpu_temperature_c = sample.temperature_c
    state.gpu_power_w = sample.power_w
    if sample.utilization is not None:
        history = _GPU_UTILIZATION_HISTORY[sample.index]
        history.append(sample.utilization)
        state.gpu_utilization_avg = sum(history) / len(history)
        state.gpu_utilization_max = max(history)


def _nvml_value(read: Callable[[], object]) -> object | None:
    try:
        return read()
    except Exception:
        return None


def _parse_float(value: str) -> float | None:
    normalized = value.strip()
    if not normalized or normalized.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _join_monitor_errors(errors: list[str]) -> str:
    unique = tuple(dict.fromkeys(error for error in errors if error))
    if not unique:
        return "GPU 监控不可用：未检测到可用监控接口"
    return "GPU 监控不可用：" + "；".join(unique)


def _monitor_error_detail(detail: str) -> str:
    if not detail:
        return ""
    if "no new privileges" in detail:
        return "当前 sandbox 禁止 sudo 提权，进程内 host-exec 桥接不可用"
    if "couldn't communicate with the NVIDIA driver" in detail:
        return "无法与 NVIDIA 驱动通信"
    if "Failed to initialize NVML" in detail:
        return "NVML 初始化失败"
    return detail
