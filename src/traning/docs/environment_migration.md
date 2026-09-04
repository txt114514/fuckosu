# environment 内迁说明

## 权威位置

环境检测的唯一实现位于 `traning.lib.environment`：

- `env_check.py` 负责依赖目录和完整报告编排；
- `gpu.py` 延迟导入 PyTorch，采集 Torch/CUDA、设备、显存、Compute Capability、cuDNN 和 BF16；
- `report.py` 负责稳定的 JSON-safe 输出；
- `training.py` 负责已验证配置与坐标/设备兼容检查；
- `check_gpu.sh` 是真实容器 namespace 的权威 GPU 脚本；
- `traning.state.environment` 是报告 DTO 的唯一类型定义位置。

`src/start/checks/registry.py` 和 `traning.core.app` 只导入新路径。仓库根 `environment` 仅保留 deprecated Python re-export、README 跳转和 shell 参数/退出码转发，不保留第二份检测逻辑。

## 报告契约

完整报告包含 Python 版本与解释器、平台、FFmpeg 路径、PyTorch/TorchVision 版本、PyTorch CUDA build、`torch.cuda.is_available()`、GPU 名称、总/空闲显存、Compute Capability、cuDNN、BF16，以及 OpenCV、PyAV、NumPy、SciPy、Pillow、Typer 和 Pydantic Settings 可用性。包检查由规格表驱动，不用并列硬编码分支。

CUDA 不可用时，报告显式返回 `cuda_available=false`、不可用原因和 `cpu_mode_allowed`。CPU 是否允许取决于调用方配置；`require_cuda=true` 时不可伪装为 warning 或自动回退。Torch 导入失败和驱动探测失败作为 typed 报告内容返回，collector 本身仍可诊断其他依赖。

## CLI 与启动接入

标准入口为：

```bash
PYTHONPATH=src python -m traning env-check --config configs/traning.yaml --no-strict
PYTHONPATH=src python -m traning.core.app.cli env-check --config configs/traning.yaml --no-strict
PYTHONPATH=src python -m start env-check --config configs/traning.yaml --no-strict
```

`start` 的宿主检查消费同一个 `EnvironmentReport`，不会在 details 阶段再次直接调用 `torch.cuda.is_available()`。配置/坐标环境检查复用已采集的报告，并将结果附加到同一报告契约。

GPU 实机检查按工程约定从正常容器 namespace 运行：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc \
  'cd /home/dev/workspace && bash src/traning/lib/environment/check_gpu.sh'
```

本迁移不更换 CUDA、PyTorch、Docker 基础镜像或 Dev Container，不安装或重装 PyTorch。
