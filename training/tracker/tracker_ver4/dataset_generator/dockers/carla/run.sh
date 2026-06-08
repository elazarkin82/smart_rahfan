#!/bin/bash
# run.sh - Start CARLA Simulator Server Docker container with GPU support
docker run -d --gpus all -p 2000-2002:2000-2002 --name carla-server-instance carla-server:local /bin/bash CarlaUE4.sh -opengl
