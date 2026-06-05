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

# Copy the packages folder cloned by build.sh
COPY rknn-toolkit2/rknn-toolkit2/packages/ /tmp/rknn_packages/

# Dynamically detect python version and install the matching RKNN-Toolkit2 wheel
RUN pip install /tmp/rknn_packages/rknn_toolkit2-*-$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")-*.whl && \
    rm -rf /tmp/rknn_packages

# Install tf2onnx at the end of the Dockerfile to preserve build cache, ensuring numpy remains in version 1.x
RUN pip install --no-cache-dir tf2onnx "numpy<2"

# Set working directory
WORKDIR /workspace
