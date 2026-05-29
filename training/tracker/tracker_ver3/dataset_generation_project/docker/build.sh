#!/bin/bash
# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define the image tag representing cv2 with nonfree on ubuntu24
IMAGE_TAG="dataset-generator-cv2-ubuntu24:latest"

echo "=================================================="
echo "Building Docker Image: ${IMAGE_TAG}"
echo "Build context directory: ${SCRIPT_DIR}"
echo "=================================================="

# Run docker build
docker build -t "${IMAGE_TAG}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"

echo "=================================================="
echo "Successfully built Docker Image: ${IMAGE_TAG}"
echo "=================================================="
