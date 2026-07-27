# 轻量 GPU 实验镜像；CUDA/cuDNN 版本由官方 PyTorch 基础镜像统一提供。
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

RUN pip3 install pandas openpyxl matplotlib

RUN apt-get update &&\
    apt-get install -y git
