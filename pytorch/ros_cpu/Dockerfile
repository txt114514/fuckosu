# ROS 2 CPU 视觉实验镜像；与主训练 devcontainer 相互独立。
#FROM elainasuki/ros:ros2-humble-full-v3
FROM elainasuki/ros:ros2-humble-full-0614

ARG USERNAME=Elaina

ARG USER_GID=$USER_UID

ARG GROUP_NAME=wheel
RUN apt-get update \
    && apt-get install -y  sudo vim nautilus 
USER ${USERNAME}
# 安装 CPU 版 PyTorch，确保 ROS 镜像不会隐式依赖 CUDA runtime。
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装yolo
RUN pip install imutils ultralytics openvino
#安装其他依赖

RUN pip install jupyter d2l==0.17.6   

USER root
RUN  echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.bashrc
ENV PYTHONPATH="/home/Elaina/yolo:${PYTHONPATH}"
