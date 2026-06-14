USER_ID=$(id -u)
USER_GROUP=$(id -g)

ADD_USER_COMMAND="groupdel users && groupadd -g ${USER_GROUP} users && useradd -m -u ${USER_ID} -g ${USER_GROUP} ${USER} && usermod --shell /bin/bash ${USER}"
VOLUMES="\
	-v ${HOME}:${HOME} \
"

docker run --rm --gpus all --cpus=16 --ipc=host --ulimit memlock=10737418240 --ulimit stack=67108864 -m 80g --memory-swap 80g --net=host --env="DISPLAY" ${VOLUMES} -it smart-rahfan-keras-gpu:latest bash -c "${ADD_USER_COMMAND} && bash"
