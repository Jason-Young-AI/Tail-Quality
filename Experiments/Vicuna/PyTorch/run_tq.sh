#!/bin/bash

source ./vars.sh

source $CONDA_PREFIX/bin/activate ${THIS_ENV_NAME}

cd ./sources

HF_HOME=${THIS_MODEL_DIR} python run_tq.py \
  --min-run 100 \
  --results-basepath ${THIS_RESULTS_DIR} \
  --warm-run 2 \
  --window-size 5 \
  --fit-run-number 1 \
  --rJSD-threshold 0.02 \
  --max-run 1000000 \
  --batch-size 1 \
  --dataset-path ${THIS_DATASET_DIR} \
  --local \
  --model-path lmsys/vicuna-13b-v1.3 \
  --run-mode val

cd ..