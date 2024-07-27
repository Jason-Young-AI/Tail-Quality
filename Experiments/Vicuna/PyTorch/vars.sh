#!/bin/bash

get_absolute_path() {
  local relative_path="$1"
  echo "$(cd "$(dirname "$relative_path")"; pwd)/$(basename "$relative_path")"
}

HF_LINK=https://huggingface.co/datasets/AIJasonYoung/Tail-Quality-Assets

THIS_MODEL_NAME=Vicuna

THIS_FRAMEWORK=PyTorch

TQ_ASSETS_ROOT=$(get_absolute_path "../../../../TQAssets")

THIS_ASSETS_DIR=${TQ_ASSETS_ROOT}/${THIS_MODEL_NAME}

THIS_DATASET_URL=${HF_LINK}/resolve/main/${THIS_MODEL_NAME}/datasets.zip
THIS_DATASET_DIR=${THIS_ASSETS_DIR}/datasets

THIS_MODEL_DIR=${THIS_ASSETS_DIR}/${THIS_FRAMEWORK}

mkdir -p ${THIS_MODEL_DIR}


TQ_RESULTS_ROOT=$(get_absolute_path "../../../../TQResults")
THIS_RESULTS_DIR=${TQ_RESULTS_ROOT}/${THIS_MODEL_NAME}_${THIS_FRAMEWORK}

mkdir -p ${THIS_RESULTS_DIR}

THIS_ENV_NAME=TQ_${THIS_MODEL_NAME}_${THIS_FRAMEWORK}