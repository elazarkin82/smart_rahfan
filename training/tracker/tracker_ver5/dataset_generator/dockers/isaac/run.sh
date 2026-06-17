#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${ISAAC_CONTAINER_NAME:-isaac-sim-instance}"
IMAGE_NAME="${ISAAC_IMAGE_NAME:-isaac-sim-server:local}"
DISPLAY_VALUE="${DISPLAY:-:1}"

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
        echo "Container '${CONTAINER_NAME}' is already running."
        exit 0
    fi
    echo "Container '${CONTAINER_NAME}' already exists but is stopped. Remove it before retrying." >&2
    exit 1
fi

# Keep the container idle. isaac_dataset_generator.py launches the only Kit
# process through SimulationApp after connect.sh prepares the runtime user.
docker run -d --rm \
    --gpus all \
    --ipc=host \
    --net=host \
    -e ACCEPT_EULA=Y \
    -e DISPLAY="${DISPLAY_VALUE}" \
    -v "${HOME}:${HOME}" \
    --entrypoint /bin/bash \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}" \
    -lc 'exec sleep infinity'
