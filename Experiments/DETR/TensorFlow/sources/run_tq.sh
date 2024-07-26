#!/bin/bash
python run_tq.py \
  --min-run 100 \
  --results-basepath /home/zxyang/HybridNets/Results \
  --warm-run 1 \
  --window-size 5 \
  --fit-run-number 2 \
  --rJSD-threshold 0.005 \
  --max-run 1000000 \
  --dataset-path /home/zxyang/1.Datasets/MS-COCO \
  --model-path /home/zxyang/HybridNet/DETR/detr-r101-dc5-a2e86def.h5