# Environment Utilities

本目录保存仓库运行环境相关代码，独立于 `src/traning` 的训练核心实现。

- `env_check.py`：收集 Python、PyTorch/CUDA、GPU、FFmpeg 和关键依赖状态。
- `check_gpu.sh`：在正确容器 namespace 中验证 `nvidia-smi` 和 PyTorch CUDA 可用性。

GPU 命令优先使用主机桥：

```bash
host-exec docker exec -u dev osu_ai_dev bash -lc \
  'cd /home/dev/workspace && bash environment/check_gpu.sh'
```

训练 CLI 仍保留环境检查命令：

```bash
PYTHONPATH=src python -m traning.cli env-check --strict --require-cuda
```
