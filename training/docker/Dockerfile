# NVIDIA TensorFlow 24.04 image (TensorFlow 2.15.0, Python 3.10)
FROM nvcr.io/nvidia/tensorflow:24.04-tf2-py3

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

# Upgrade pip and install OpenCV (cv2)
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "numpy<2" opencv-python tqdm

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

CMD ["/bin/bash"]
