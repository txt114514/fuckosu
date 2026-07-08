FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DISPLAY=:0

# 系统依赖
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        git wget curl vim nano htop \
        build-essential pkg-config \
        ffmpeg \
        tmux \
        libevdev-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libx11-6 libxrandr2 libxi6 libxtst6 libxfixes3 \
        libxcursor1 libxinerama1 libxxf86vm1 \
        libnss3 libatk1.0-0 libatk-bridge2.0-0 \
        libgbm1 libasound2 \
        iputils-ping curl sudo util-linux \
        x11-utils mesa-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 创建用户 dev 并加入 sudo 组
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN if ! getent group ${USER_GID} >/dev/null; then \
        groupadd -g ${USER_GID} ${USERNAME}; \
    fi && \
    useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME}

COPY bin/host-exec /usr/local/bin/host-exec
RUN chmod 0755 /usr/local/bin/host-exec && \
    echo "${USERNAME} ALL=(root) NOPASSWD: /usr/local/bin/host-exec" > /etc/sudoers.d/host-exec && \
    chmod 0440 /etc/sudoers.d/host-exec && \
    visudo -cf /etc/sudoers.d/host-exec

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# pip 全局安装
RUN pip install --no-cache-dir \
    numpy scipy pandas matplotlib openpyxl pillow \
    opencv-python-headless \
    tqdm mss dxcam \
    pyautogui python-xlib evdev \
    torchvision torchaudio \
    stable-baselines3 gymnasium \
    tensorboard slider \
    rich loguru \
    typer \
    prefect \
    pydantic \
    pydantic-settings \
    pyyaml \
    sqlmodel \
    alembic \
    av \
    einops \
    safetensors \
    psutil \
    optuna \
    pytest \
    pytest-cov \
    mypy \
    types-PyYAML \
    pathspec \
    ruff
CMD ["/bin/bash"]
