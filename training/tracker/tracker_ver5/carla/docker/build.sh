#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the image name and tag
IMAGE_NAME="smart-rahfan-carla"
TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}..."
echo "Using build context: ${SCRIPT_DIR}"

# Build the Docker image
docker build -t "${IMAGE_NAME}:${TAG}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"

echo "Docker image built successfully: ${IMAGE_NAME}:${TAG}"
