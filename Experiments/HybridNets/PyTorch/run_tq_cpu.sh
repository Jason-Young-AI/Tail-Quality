#!/bin/bash

source ./vars.sh

source $CONDA_PREFIX/bin/activate ${THIS_ENV_NAME}

cd ./sources

CUDA_VISIBLE_DEVICES="" python run_tq.py \
  --min-run 100 \
  --results-basepath ${THIS_RESULTS_DIR} \
  --warm-run 5 \
  --window-size 5 \
  --fit-run-number 2 \
  --rJSD-threshold 0.02 \
  --max-run 1000000 \
  --dataset-path ${THIS_DATASET_DIR} \
  --model-path ${THIS_MODEL_DIR}/${THIS_MODEL_FILENAME} \
  --num_gpus 0

cd ..