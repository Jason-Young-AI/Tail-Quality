#!/bin/bash

source ./vars.sh

source $CONDA_PREFIX/bin/activate ${THIS_ENV_NAME}

cd ./sources

mkdir -p ${THIS_RESULTS_DIR}/GPU

python run_tq.py \
  --results-basepath ${THIS_RESULTS_DIR}/GPU \
  --warm-run 30 \
  --window-size 5 \
  --fit-run-number 6 \
  --batch-size 100 \
  --rJSD-threshold 0.02 \
  --max-run 1000000 \
  --dataset-path ${THIS_DATASET_DIR} \
  --model-path ${THIS_MODEL_DIR}/${THIS_MODEL_FILENAME}

cd ..
