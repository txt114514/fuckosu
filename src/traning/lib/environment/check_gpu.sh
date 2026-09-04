#!/usr/bin/env bash
# 在真实运行 namespace 中只读验证驱动、PyTorch CUDA、BF16 与显存查询。
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is unavailable in this process namespace" >&2
fi

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="${workspace_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m traning.lib.environment.gpu --require-cuda
