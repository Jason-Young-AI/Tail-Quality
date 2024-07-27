#!/bin/bash
HF_HOME=/home/zxyang/HybridNet/cache/hub python run_tq.py \
  --min-run 100 \
  --results-basepath ../Results \
  --warm-run 1 \
  --window-size 5 \
  --fit-run-number 2 \
  --rJSD-threshold 0.005 \
  --max-run 1000000 \
  --batch-size 1 \
  --dataset-path /home/zxyang/HybridNet/data/data \
  --local \
  --model-path lmsys/vicuna-13b-v1.3 \
  --run-mode val