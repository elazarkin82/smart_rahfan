#!/usr/bin/env bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The project root is two levels up from rknn/docker/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"

USER_ID=$(id -u)
USER_GROUP=$(id -g)

ADD_USER_COMMAND="groupdel users && groupadd -g ${USER_GROUP} users && useradd -m -u ${USER_ID} -g ${USER_GROUP} ${USER} && usermod --shell /bin/bash ${USER}"
VOLUMES="\
	-v /home/elazarkin/programs:/home/elazarkin/programs \
	-v $HOME/.Xauthority:$HOME/.Xauthority:rw \
	-v /home/elazarkin/work/deeplearning/home:/home/elazarkin \
	-v /home/elazarkin/.cache:/home/elazarkin/.cache \
	-v /home/elazarkin/.p2:/home/elazarkin/.p2 \
	-v /home/elazarkin/work/media/camera_sort:/home/elazarkin/work/media/camera_sort \
	-v /home/elazarkin/work/libs:/home/elazarkin/work/libs \
	-v /home/elazarkin/Desktop/work:/home/elazarkin/Desktop/work \
	-v /home/elazarkin/storage/datasets:/home/elazarkin/storage/datasets \
	-v /home/elazarkin/storage/video_samples:/home/elazarkin/storage/video_samples \
	-v /home/elazarkin/storage/private:/home/elazarkin/storage/private:ro \
	-v ${PROJECT_ROOT}:${PROJECT_ROOT} \
"

docker run --rm --gpus all --cpus=8 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -m 20g --net=host --env="DISPLAY" --workdir="${PROJECT_ROOT}" ${VOLUMES} -it smart-rahfan-rknn:latest bash -c "${ADD_USER_COMMAND} && bash"
