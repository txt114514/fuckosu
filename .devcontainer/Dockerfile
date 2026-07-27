# 主开发容器：CUDA/PyTorch 基础镜像上补齐训练、视频、X11 与中文 UI 依赖。
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
        libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libnss3 libatk1.0-0 libatk-bridge2.0-0 \
        libgbm1 libasound2 \
        fontconfig fonts-noto-cjk \
        iputils-ping curl sudo util-linux \
        x11-utils mesa-utils && \
    fc-cache -f && \
    fc-match "Noto Sans CJK SC" && \
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
# host-exec 是受限的主机桥；仅授予该固定入口免密 sudo，不开放任意 root shell。
RUN chmod 0755 /usr/local/bin/host-exec && \
    echo "${USERNAME} ALL=(root) NOPASSWD: /usr/local/bin/host-exec" > /etc/sudoers.d/host-exec && \
    chmod 0440 /etc/sudoers.d/host-exec && \
    visudo -cf /etc/sudoers.d/host-exec

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# pip 全局安装
COPY requirements.txt /tmp/osu-ai-requirements.txt
RUN pip install --no-cache-dir -r /tmp/osu-ai-requirements.txt
CMD ["/bin/bash"]
