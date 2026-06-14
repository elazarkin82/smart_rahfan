# NVIDIA TensorFlow 25.01 image (TensorFlow 2.17.0, Python 3.12, CUDA 12.8)
FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
# python3-tk is required for Tkinter (tk) GUI toolkit
# libgl1 and libglib2.0-0 are required for OpenCV (cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install OpenCV (cv2) and model optimization
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "numpy<2" opencv-python tqdm tensorflow-model-optimization
    
RUN python3 -m pip install pillow

RUN apt-get update && apt-get install -y --no-install-recommends geany

# Build-time sanity check
RUN python3 - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices("GPU"))

import tkinter as tk
print("Tkinter (tk) successfully loaded. TkVersion:", tk.TkVersion)

import cv2
print("OpenCV (cv2) successfully loaded. Version:", cv2.__version__)

import tqdm
print("tqdm successfully loaded. Version:", tqdm.__version__)
PY

# Install dependencies for CARLA tracker_ver4 dataset_generator
# carla is only installed on x86_64 / amd64 since pre-compiled wheels are not available for arm64.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        python3 -m pip install pygame carla==0.9.16; \
    else \
        python3 -m pip install pygame; \
    fi

CMD ["/bin/bash"]

