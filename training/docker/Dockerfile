# TensorFlow 2.18 GPU image
FROM tensorflow/tensorflow:2.18.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV TF_CPP_MIN_LOG_LEVEL=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    wget \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    xterm \
    poppler-utils \
    bash-completion \
    openjdk-17-jdk \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Bash completion
RUN echo "if [ -f /etc/bash_completion ]; then . /etc/bash_completion; fi" >> /etc/bash.bashrc

# Upgrade pip tooling
RUN python3 -m pip install --upgrade pip setuptools wheel

# TensorFlow 2.18-compatible pins
RUN python3 -m pip install \
    "numpy>=1.26.0,<2" \
    "protobuf>=3.20.3,<6" \
    "tf-keras==2.18.*" \
    "tensorflow-probability==0.25.0"

# Utility / ONNX packages
# onnxruntime here is CPU runtime. Safer than mixing another GPU stack into the TF image.
RUN python3 -m pip install \
    "onnx==1.17.0" \
    "onnxruntime==1.19.0" \
    "onnxsim==0.4.36" \
    "opencv-python-headless==4.10.0.84" \
    tqdm \
    netron \
    pdf2image

# Build-time sanity check
RUN python3 - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

import keras
print("keras:", keras.__version__)

import tf_keras
print("tf_keras:", tf_keras.__version__)

import tensorflow_probability as tfp
print("tensorflow_probability:", tfp.__version__)

import onnx
print("onnx:", onnx.__version__)

import onnxruntime as ort
print("onnxruntime:", ort.__version__)
print("onnxruntime providers:", ort.get_available_providers())

import cv2
print("opencv:", cv2.__version__)
PY

CMD ["/bin/bash"]
