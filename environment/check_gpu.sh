#!/usr/bin/env bash
set -euo pipefail

nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the container")

print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("bf16:", torch.cuda.is_bf16_supported())

free_bytes, total_bytes = torch.cuda.mem_get_info()
print("free_vram_gib:", free_bytes / 1024**3)
print("total_vram_gib:", total_bytes / 1024**3)
PY
