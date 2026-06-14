USER_ID=$(id -u)
USER_GROUP=$(id -g)

ADD_USER_COMMAND="groupdel users && groupadd -g ${USER_GROUP} users && useradd -m -u ${USER_ID} -g ${USER_GROUP} ${USER} && usermod --shell /bin/bash ${USER}"
VOLUMES="\
	-v ${HOME}:${HOME}
"

sudo docker run --rm --cpus=16 -e CUDA_VISIBLE_DEVICES=-1 --net=host --env="DISPLAY" ${VOLUMES} -it local_ubuntu_24_4 bash -c "${ADD_USER_COMMAND} && bash"
