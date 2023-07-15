#!/bin/bash

ACTION=${1}
THIS_PATH=$(cd $(dirname $0); pwd)

#VERSION=${2}

DOCKER_CON_NAME="Serving_DETR_TF"

if [[ $ACTION == "start" ]]; then
	#CUDA_VISIBLE_DEVICES=0 docker run --runtime=nvidia --gpus '"device=0"' --name ${DOCKER_CON_NAME} -p 8500:8500 -p 8501:8501 \
	#docker run --gpus '"device=0"' --name ${DOCKER_CON_NAME} -p 8500:8500 -p 8501:8501 \
	#--mount type=bind,source="${THIS_PATH}/bins/jx_vit_large_p32_tf2.tf",target=/models/ViT_L_P32_384 \
	#-e NVIDIA_DISABLE_REQUIRE=1 \
	#-e TF_CPP_MIN_VLOG_LEVEL=4 \
	#-e MODEL_NAME=ViT_L_P32_384 -t tensorflow/serving:latest-gpu &

	#CUDA_VISIBLE_DEVICES=0 docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --gpus '"device=0"' --name ${DOCKER_CON_NAME} -p 8500:8500 -p 8501:8501 \
	#docker network create zxyang_net
	#docker run --runtime=nvidia --gpus '"device=0"' --net zxyang_net --name ${DOCKER_CON_NAME} -p 8500:8500 -p 8501:8501 \
	docker run --runtime=nvidia --gpus '"device=0"' --name ${DOCKER_CON_NAME} -p 8500:8500 -p 8501:8501 \
		--mount type=bind,source="${THIS_PATH}/bins/detr-r101-dc5-a2e86def.h5",target=/models/DETR_ResNet101 \
		--mount type=bind,source="${THIS_PATH}/configs",target=/configs \
		-v ${THIS_PATH}/tensorboard:/tmp/tensorboard \
		-e MODEL_NAME=DETR_ResNet101 -t tensorflow/serving:2.11.0-gpu \
		--model_config_file=/configs/models.config \
		--monitoring_config_file=/configs/monitor.config &
		#-e MODEL_NAME=ViT_L_P32_384 -t zxyang/tensorflow-serving-gpu:latest &
		#--mount type=bind,source="${THIS_PATH}/configs",target=/configs \
		#--model_config_file=/configs/models.config \
		#--monitoring_config_file=/configs/monit.config &
	echo "Container ${DOCKER_CON_NAME} Created!"
	#if [[ ${VERSION} == "BSZ1" ]]; then
	#	CUDA_VISIBLE_DEVICES=0 torchserve --start --model-store model_store --ts-config ./ts_config_bsz1.properties
	#	echo "Started"
	#elif [[ ${VERSION} == "BSZ2" ]]; then
	#	CUDA_VISIBLE_DEVICES=0 torchserve --start --model-store model_store --ts-config ./ts_config_bsz2.properties
	#	echo "Started"
	#else
	#	echo "Not Support Version: ${VERSION}"
	#fi
elif [[ $ACTION == "stop" ]]; then
	#docker ps -a -q  --filter ancestor=tensorflow/serving:latest-gpu  --format="{{.ID}}"
	#docker rm $(docker stop $(docker ps -a -q --filter="name=${DOCKER_CON_NAME}"))
	CON_ID=$(docker ps -a -q --filter="name=${DOCKER_CON_NAME}")
	docker stop ${CON_ID} || true && docker rm ${CON_ID} || true
	#docker network rm zxyang_net
	echo "Stoped"
else
	echo "Wrong Command"
fi
