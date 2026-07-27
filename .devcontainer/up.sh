#!/usr/bin/env bash

# 启动或复用本地开发容器，并在需要时配置 X11 访问权限。

set -euo pipefail

# 1. 确保 xhost 允许本地 docker 容器访问 X server。
#    安全边界：该授权不会由脚本自动撤销，仅应在可信的本机开发环境使用。
if command -v xhost >/dev/null 2>&1; then
    xhost +local:docker || true   # 如果已经设置过，不会报错
else
    echo "Warning: xhost not found, skipping X11 permission"
fi

# 2. 可选：如果你的 DISPLAY 不是 :0，自动设置（很少需要）
export DISPLAY=${DISPLAY:-:0}

# 3. 启动 compose。默认复用本地镜像，避免每次启动都触发基础镜像 pull。
IMAGE_NAME=${DEVCONTAINER_IMAGE:-osu-ai-dev:latest}
COMPOSE_UP_ARGS=(-d)

if [[ "${FORCE_BUILD:-0}" == "1" ]]; then
    echo "FORCE_BUILD=1，重新构建镜像（仅缺失时拉取基础镜像）..."
    COMPOSE_UP_ARGS+=(--build --pull missing)
elif docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "发现本地镜像 $IMAGE_NAME，跳过构建和拉取。"
    COMPOSE_UP_ARGS+=(--no-build --pull never)
else
    echo "未发现本地镜像 $IMAGE_NAME，开始构建（仅缺失时拉取基础镜像）..."
    COMPOSE_UP_ARGS+=(--build --pull missing)
fi

docker compose up "${COMPOSE_UP_ARGS[@]}" "$@"

echo ""
echo "容器已启动。"
echo "进入容器：docker compose exec osu-dev bash"
echo "测试 X11：容器内运行 xeyes 或 python -c 'import pyautogui; print(pyautogui.position())'"
echo "测试 GPU：容器内运行 nvidia-smi"
