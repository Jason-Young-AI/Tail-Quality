#!/bin/bash

source ./vars.sh

source $CONDA_PREFIX/bin/activate ${THIS_ENV_NAME}

cd ./sources

python run_tq.py \
  --min-run 100 \
  --results-basepath ${THIS_RESULTS_DIR} \
  --warm-run 1 \
  --window-size 5 \
  --fit-run-number 1 \
  --rJSD-threshold 0.02 \
  --max-run 1000000 \
  --dataset-path ${THIS_DATASET_DIR} \
  --model-path ${THIS_MODEL_DIR}/${THIS_MODEL_FILENAME} \
  --only-quality \
  --golden-path ${THIS_RESULTS_DIR}/golden.json \
  --result-path ${THIS_RESULTS_DIR}/result.json \
  --others-path ${THIS_RESULTS_DIR}/others.json

cd ..
