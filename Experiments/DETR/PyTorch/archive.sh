#!/bin/bash
if [ $# -ne 1 ]; then
    echo "Usage ./archive.sh [ BSZ1 | BSZ2 ]"
    exit 1
fi

VERSION=${1}

if [[ ${VERSION} != "BSZ1" && ${VERSION} != "BSZ2" ]]; then
	echo "Not Support Version: ${VERSION}"
	exit 1
fi

SOURCE_DIR=""

if [[ ${VERSION} == "BSZ1" ]]; then
    SOURCE_DIR="with_dilation_aka_bsz1"
fi

if [[ ${VERSION} == "BSZ2" ]]; then
    SOURCE_DIR="without_dilation_aka_bsz2"
fi



CUDA_VISIBLE_DEVICES=0 torch-model-archiver \
    --model-name DETR_ResNet101_${VERSION} \
    --version 1.0 \
    --model-file sources/${SOURCE_DIR}/detr.py \
    --extra-files sources/${SOURCE_DIR}/backbone.py,sources/${SOURCE_DIR}/transformer.py,sources/${SOURCE_DIR}/mlp.py,sources/${SOURCE_DIR}/position_encoding.py,sources/${SOURCE_DIR}/utils.py,sources/${SOURCE_DIR}/misc.py,sources/${SOURCE_DIR}/model.hocon,sources/${SOURCE_DIR}/box_ops.py,sources/${SOURCE_DIR}/transforms.py \
    --serialized-file ./bins/DETR_ResNet101_${VERSION}.pth \
    --handler sources/${SOURCE_DIR}/detr_handler
    #--config-file model_config.json

mkdir -p model_store
mv DETR_ResNet101_${VERSION}.mar model_store
