#!/bin/bash
if [ $# -ne 2 ]; then
    echo "Usage ./archive.sh [ start | stop ] [ BSZ1 | BSZ2 ]"
    exit 1
fi

ACTION=${1}

VERSION=${2}

if [[ $ACTION == "start" ]]; then
	if [[ ${VERSION} == "BSZ1" ]]; then
		CUDA_VISIBLE_DEVICES=0 torchserve --start --model-store model_store --ts-config ./ts_config_bsz1.properties
		echo "Started"
	elif [[ ${VERSION} == "BSZ2" ]]; then
		CUDA_VISIBLE_DEVICES=0 torchserve --start --model-store model_store --ts-config ./ts_config_bsz2.properties
		echo "Started"
	else
		echo "Not Support Version: ${VERSION}"
	fi
elif [[ $ACTION == "stop" ]]; then
	torchserve --stop
	echo "Stoped"
else
	echo "Wrong Command"
fi
