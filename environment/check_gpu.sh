#!/usr/bin/env bash
# 已弃用的兼容入口；真实实现位于训练包内部。
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${workspace_root}/src/traning/lib/environment/check_gpu.sh" "$@"
