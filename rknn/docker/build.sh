#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the image name and tag
IMAGE_NAME="smart-rahfan-rknn"
TAG="latest"

# Clone the official RKNN-Toolkit2 repository if not already cloned
if [ ! -d "${SCRIPT_DIR}/rknn-toolkit2" ]; then
    echo "Cloning RKNN-Toolkit2 repository (includes packages & wheels)..."
    git clone https://github.com/rockchip-linux/rknn-toolkit2.git "${SCRIPT_DIR}/rknn-toolkit2"
else
    echo "RKNN-Toolkit2 repository already exists, skipping clone."
fi

echo "Building Docker image: ${IMAGE_NAME}:${TAG}..."
echo "Using build context: ${SCRIPT_DIR}"

# Build the Docker image using the script's directory as context
docker build -t "${IMAGE_NAME}:${TAG}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"

echo "Docker image built successfully: ${IMAGE_NAME}:${TAG}"
