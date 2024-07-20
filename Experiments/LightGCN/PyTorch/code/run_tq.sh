#!/bin/bash
python run_tq.py \
  --min-run-number 100 \
  --results-basepath ../../../../Results \
  --warm-run 1 \
  --window-size 5 \
  --fit-batches 100 \
  --rJSD-threshold 0.005