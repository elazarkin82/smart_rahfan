#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}/dockers/isaac"
IMAGE_NAME="${ISAAC_IMAGE_NAME:-isaac-sim-server:local}"
CONTAINER_NAME="${ISAAC_CONTAINER_NAME:-isaac-sim-instance}"

force_rebuild=false
keep_container=false
generator_args=()
started_container=false

usage() {
    cat <<'EOF'
Usage: ./run_isaac_datagenerator.sh [options] [-- generator arguments]

Builds the Isaac Sim image when needed, starts the Docker container, and runs
isaac_dataset_generator.py to populate the configured cache directory.

Options:
  --rebuild          Rebuild the Isaac Sim Docker image before generation.
  --keep-container   Keep the container running after generation.
  -h, --help         Show this help message.

Environment:
  ISAAC_IMAGE_NAME       Docker image name (default: isaac-sim-server:local).
  ISAAC_CONTAINER_NAME   Container name (default: isaac-sim-instance).
  DISPLAY                Display used by Isaac Sim (default: :1).
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --rebuild)
            force_rebuild=true
            shift
            ;;
        --keep-container)
            keep_container=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            generator_args=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cleanup() {
    local exit_code=$?

    if [[ "${started_container}" == "true" && "${keep_container}" != "true" ]]; then
        echo "[*] Stopping ${CONTAINER_NAME}..."
        docker stop "${CONTAINER_NAME}" >/dev/null || true
    fi

    exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed or is not available in PATH." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is unavailable or the current user lacks Docker access." >&2
    exit 1
fi

if [[ "${force_rebuild}" == "true" ]] || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "[*] Building Isaac Sim image: ${IMAGE_NAME}"
    (
        cd "${DOCKER_DIR}"
        docker build -t "${IMAGE_NAME}" .
    )
else
    echo "[+] Isaac Sim image already exists: ${IMAGE_NAME}"
fi

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
        echo "Container '${CONTAINER_NAME}' exists but is stopped. Remove it and retry." >&2
        exit 1
    fi
    echo "[+] Reusing running container: ${CONTAINER_NAME}"
else
    echo "[*] Starting Isaac Sim container: ${CONTAINER_NAME}"
    ISAAC_CONTAINER_NAME="${CONTAINER_NAME}" \
        ISAAC_IMAGE_NAME="${IMAGE_NAME}" \
        "${DOCKER_DIR}/run.sh"
    started_container=true
fi

echo "[*] Generating Isaac dataset cache..."
ISAAC_CONTAINER_NAME="${CONTAINER_NAME}" \
    "${DOCKER_DIR}/connect.sh" \
    /isaac-sim/python.sh \
    isaac_dataset_generator.py \
    "${generator_args[@]}"

cache_count="$(find "${SCRIPT_DIR}/cache" -maxdepth 1 -type f -name 'flight_*.pkl' 2>/dev/null | wc -l)"
echo "[+] Isaac cache generation finished. Cache files: ${cache_count}"
