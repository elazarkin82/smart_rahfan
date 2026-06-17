#!/bin/bash
# run.sh - Start CARLA Simulator Server Docker container with GPU support
# docker run -d --gpus all -p 2000-2002:2000-2002 --name carla-server-instance carla-server:local /bin/bash CarlaUE4.sh -opengl

# Run the customized carla image
docker run --rm -it \
    --name carla_server \
    --net=host \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    ${VOLUMES} \
    carla-server:local \
    /bin/bash ./CarlaUE4.sh -vulkan -RenderOffScreen -quality-level=Low
    
#/bin/bash ./CarlaUE4.sh -vulkan -quality-level=Low -RenderOffScreen
