#!/bin/bash
# V100
## LLM
python calculate_stats.py --data-dir ../Results/Raw/V100_LLM_PyTorch_val_bsz1/ --data-filename LLM_Run100 --batch-size 1 --threshold 0.2 --tolerance 5 --init-num 30 --check-npz-path ./temp.npz  --fit-type kde --n-level specific --detect-type cumulate --dataset-type MMLU --combine-type i --rm-outs-type none --step 5 --quality-type acc --check-min-n