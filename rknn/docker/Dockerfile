# Inherit from the local smart-rahfan-keras-gpu image
FROM smart-rahfan-keras-gpu:latest

# Switch to root to install system packages
USER root

# Install system dependencies required by RKNN-Toolkit2 and ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    gcc \
    g++ \
    build-essential \
    libxslt1-dev \
    zlib1g-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libprotobuf-dev \
    protobuf-compiler \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install standard python dependencies for RKNN-Toolkit2
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.22.4 \
    onnx==1.12.0 \
    onnxruntime==1.12.1 \
    opencv-python \
    pyyaml \
    protobuf==3.19.4 \
    flatbuffers==2.0 \
    tqdm

# Document how to install RKNN-Toolkit2 Wheel:
# 1. Download the rknn-toolkit2 repository: git clone https://github.com/rockchip-linux/rknn-toolkit2.git
# 2. Find the wheel file matching your python version (e.g., cp310 for Python 3.10) in the doc/wheel/ directory
# 3. Copy the wheel file to the same folder as this Dockerfile and uncomment the following lines:
#
# COPY rknn_toolkit2-2.0.0b0-cp310-cp310-linux_x86_64.whl /tmp/
# RUN pip install /tmp/rknn_toolkit2-*.whl && rm /tmp/rknn_toolkit2-*.whl

# Set working directory
WORKDIR /workspace
