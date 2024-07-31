#!/bin/bash

source ./vars.sh

source $CONDA_PREFIX/bin/activate ${THIS_ENV_NAME}

cd ./sources

python run_tq.py \
  --results-basepath ${THIS_RESULTS_DIR} \
  --warm-run 1 \
  --device gpu \
  --window-size 5 \
  --fit-run-number 2 \
  --batch-size 100 \
  --rJSD-threshold 0.005 \
  --max-run 1000000 \
  --dataset-path ${THIS_DATASET_DIR} \
  --model-path /younger/peng/Tail-Quality/Experiments/MobileNet/ONNX/model/mobilenetv2-7.onnx

cd ..