#!/bin/bash
# run.sh - Start NVIDIA Isaac Sim Server Docker container in background (detached) with GPU, EULA accepted, and home mount
VOLUMES="\
	-v ${HOME}:${HOME} \
"

docker run -d --rm --gpus all -e "ACCEPT_EULA=Y" --ipc=host --net=host --env="DISPLAY" ${VOLUMES} --name isaac-sim-instance isaac-sim-server:local
