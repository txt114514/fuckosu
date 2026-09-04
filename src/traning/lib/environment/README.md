# Training Environment

该目录是环境探测的唯一实现位置。所有检查均为只读操作，不安装依赖、不切换
PyTorch/CUDA 版本，也不修改全局运行时配置。

- `report.py`：依据统一规格表收集 Python、关键包、FFmpeg、PyTorch、TorchVision、
  CUDA、GPU 显存、计算能力、cuDNN 与 BF16 状态。
- `env_check.py`：完整宿主环境报告及独立命令行入口。
- `gpu.py`：CUDA/GPU 专项结构化检查。
- `training.py`：结合 V2 配置和坐标标定证据的训练启动门禁。
- `check_gpu.sh`：在当前容器 namespace 内运行 `nvidia-smi` 和严格 CUDA 检查。

CPU 模式默认允许，但报告始终保留 `cuda_available` 和
`cuda_unavailable_reason`；要求 CUDA 时不会静默退回 CPU。

```bash
PYTHONPATH=src python -m traning.lib.environment.env_check
PYTHONPATH=src python -m traning.lib.environment.env_check --strict --require-cuda
bash src/traning/lib/environment/check_gpu.sh
```

若 Codex 的普通 sandbox 看不到 GPU，应从 devcontainer 主机桥运行：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc \
  'cd /home/dev/workspace && bash src/traning/lib/environment/check_gpu.sh'
```

仓库根目录 `environment/` 仅保留弃用兼容包装；新代码必须从
`traning.lib.environment` 导入。
