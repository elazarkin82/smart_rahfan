#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${ISAAC_CONTAINER_NAME:-isaac-sim-instance}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER="${USER:-hostuser}"
HOST_HOME="${HOME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_USER="${HOST_USER}"

if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "Container '${CONTAINER_NAME}' does not exist. Run ./run.sh first." >&2
    exit 1
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    echo "Container '${CONTAINER_NAME}' is not running. Run ./run.sh first." >&2
    exit 1
fi

docker exec -i -u root \
    -e HOST_UID="${HOST_UID}" \
    -e HOST_GID="${HOST_GID}" \
    -e HOST_USER="${HOST_USER}" \
    -e HOST_HOME="${HOST_HOME}" \
    "${CONTAINER_NAME}" bash -s <<'SETUP_USER'
set -euo pipefail

group_name="$(getent group "${HOST_GID}" | cut -d: -f1 || true)"
if [[ -z "${group_name}" ]]; then
    group_name="${HOST_USER}"
    if getent group "${group_name}" >/dev/null; then
        group_name="host-${HOST_GID}"
    fi
    groupadd --gid "${HOST_GID}" "${group_name}"
fi

if getent passwd "${HOST_USER}" >/dev/null; then
    existing_uid="$(id -u "${HOST_USER}")"
    if [[ "${existing_uid}" != "${HOST_UID}" ]]; then
        echo "User '${HOST_USER}' already exists with UID ${existing_uid}, expected ${HOST_UID}." >&2
        exit 1
    fi
else
    useradd \
        --uid "${HOST_UID}" \
        --gid "${HOST_GID}" \
        --home-dir "${HOST_HOME}" \
        --no-create-home \
        --shell /bin/bash \
        "${HOST_USER}"
fi

mkdir -p \
    "${HOST_HOME}/.cache/isaac-sim" \
    "${HOST_HOME}/.cache/ov/hub" \
    "${HOST_HOME}/.config" \
    "${HOST_HOME}/.local/share/ov" \
    /isaac-sim/kit/cache \
    /isaac-sim/.nvidia-omniverse \
    /var/cache/hub

chown -R "${HOST_UID}:${HOST_GID}" \
    "${HOST_HOME}/.cache/isaac-sim" \
    "${HOST_HOME}/.cache/ov" \
    "${HOST_HOME}/.config" \
    "${HOST_HOME}/.local/share/ov" \
    /isaac-sim/.nvidia-omniverse
chmod 1777 /isaac-sim/kit/cache /var/cache/hub
SETUP_USER

docker_tty_args=(-it)
container_command=(bash)
if (( $# > 0 )); then
    docker_tty_args=()
    container_command=("$@")
fi

exec docker exec "${docker_tty_args[@]}" \
    -u "${HOST_UID}:${HOST_GID}" \
    -w "${WORKDIR}" \
    -e HOME="${HOST_HOME}" \
    -e USER="${CONTAINER_USER}" \
    -e LOGNAME="${CONTAINER_USER}" \
    -e XDG_CACHE_HOME="${HOST_HOME}/.cache" \
    -e XDG_CONFIG_HOME="${HOST_HOME}/.config" \
    -e XDG_DATA_HOME="${HOST_HOME}/.local/share" \
    -e HUB__CACHE__PATH="${HOST_HOME}/.cache/ov/hub" \
    -e ISAAC_SIM_PORTABLE_ROOT="${HOST_HOME}/.cache/isaac-sim" \
    "${CONTAINER_NAME}" "${container_command[@]}"
