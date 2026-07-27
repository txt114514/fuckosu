# 轻量 CPU 实验镜像；固定从 PyTorch CPU wheel 源安装，避免误拉 CUDA 依赖。
FROM ubuntu:22.04
RUN apt-get update\ 
    && apt-get install -y git  python3 python3-pip 
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install pandas openpyxl matplotlib
