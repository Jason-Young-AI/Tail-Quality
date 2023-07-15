#!/bin/bash
#VERSION=${1}
#if [[ ${VERSION} != "BSZ1" && ${VERSION} != "BSZ2" ]]; then
#	echo "Not Support Version: ${VERSION}"
#	exit 1
#fi
CUDA_VISIBLE_DEVICES=0 torch-model-archiver \
    --model-name ViT_L_P32_384 \
    --version 1.0 \
    --model-file sources/vit.py \
    --extra-files sources/vit_layers.py,sources/utils.py \
    --serialized-file ./bins/jx_vit_large_p32_384-9b920ba8.pth \
    --handler sources/vit_handler

mkdir -p model_store
mv ViT_L_P32_384.mar model_store
