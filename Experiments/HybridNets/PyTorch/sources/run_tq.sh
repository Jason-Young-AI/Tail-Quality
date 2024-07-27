#!/bin/bash
python run_tq.py \
  --min-run 100 \
  --results-basepath ../Results \
  --warm-run 1 \
  --window-size 5 \
  --fit-run-number 2 \
  --rJSD-threshold 0.005 \
  --max-run 1000000 \
   -p bdd100k \
   -c 3 \
   -w weights/hybridnets.pth \
   --num_gpus 1
  #  --batch_size 1