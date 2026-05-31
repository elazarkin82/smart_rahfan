#!/usr/bin/env bash

# Define volumes
VOLUMES="\
	-v /home/elazarkin/work/projects:/home/elazarkin/work/projects \
	-v $HOME/.Xauthority:$HOME/.Xauthority:rw \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
"

echo "Configuring X11 forwarding for live viewing..."
xhost +local:root

echo "Launching CARLA Container..."
# Run the customized carla image
docker run --rm -it \
    --name carla_server \
    --net=host \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    ${VOLUMES} \
    smart-rahfan-carla:latest \
    /bin/bash ./CarlaUE4.sh -vulkan -quality-level=Low -RenderOffScreen
