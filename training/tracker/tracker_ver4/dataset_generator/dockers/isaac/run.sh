#!/bin/bash
# run.sh - Start NVIDIA Isaac Sim Server Docker container with GPU support, Omniverse EULA accepted, and user/home mapping
USER_ID=$(id -u)
USER_GROUP=$(id -g)

ADD_USER_COMMAND="groupdel users && groupadd -g ${USER_GROUP} users && useradd -m -u ${USER_ID} -g ${USER_GROUP} ${USER} && usermod --shell /bin/bash ${USER}"
VOLUMES="\
	-v ${HOME}:${HOME} \
"

docker run --rm --gpus all -e "ACCEPT_EULA=Y" --ipc=host --net=host --env="DISPLAY" ${VOLUMES} -it isaac-sim-server:local bash -c "${ADD_USER_COMMAND} && bash"
