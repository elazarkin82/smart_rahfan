#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the image name and tag
IMAGE_NAME="local_ubuntu_24_4"
TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}..."
echo "Using build context: ${SCRIPT_DIR}"

# Build the Docker image using the script's directory as context
# This avoids sending the entire project directory (which may contain large datasets) to the docker daemon.
docker build -t "${IMAGE_NAME}:${TAG}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"

echo "Docker image built successfully: ${IMAGE_NAME}:${TAG}"
