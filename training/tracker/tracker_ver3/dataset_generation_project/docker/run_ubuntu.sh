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
"

docker run --rm --cpus=12 -m 20g -e CUDA_VISIBLE_DEVICES=-1 --net=host --env="DISPLAY" ${VOLUMES} -it dataset-generator-cv2-ubuntu24:latest bash -c "${ADD_USER_COMMAND} && bash"
