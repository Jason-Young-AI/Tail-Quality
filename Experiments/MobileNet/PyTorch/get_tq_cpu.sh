#!/bin/bash

source ./vars.sh

source $CONDA_PREFIX/bin/activate ${THIS_ENV_NAME}

cd ./sources

export XDG_CACHE_HOME=${THIS_MODEL_DIR}

mkdir -p ${THIS_RESULTS_DIR}/CPU

CUDA_VISIBLE_DEVICES="" python run_tq.py \
  --results-basepath ${THIS_RESULTS_DIR}/CPU \
  --warm-run 30 \
  --window-size 5 \
  --fit-run-number 6 \
  --batch-size 100 \
  --rJSD-threshold 0.02 \
  --max-run 1000000 \
  --dataset-path ${THIS_DATASET_DIR} \
  --only-quality \
  --result-path ${THIS_RESULTS_DIR}/CPU/result.pkl \
  --golden-path ${THIS_RESULTS_DIR}/CPU/golden.pkl \
  --others-path ${THIS_RESULTS_DIR}/CPU/others.pkl \

cd ..