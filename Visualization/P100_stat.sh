#!/bin/bash
# P100
## DETR-DC5
python calculate_stats.py --data-dir ../Results/Raw/P100_DETR_TensorFlow_val_bsz1/ --data-filename DETR_Run100 --batch-size 1 --threshold 0.2 --tolerance 5 --init-num 30 --check-npz-path ../Results/stat_npzs/P100_DETR_TensorFlow_val_bsz1.npz  --fit-type kde --n-level specific --detect-type cumulate --dataset-type COCO --combine-type i --rm-outs-type none --step 5 --quality-type map --check-min-n

python calculate_stats.py --data-dir ../Results/Raw/P100_DETR_PyTorch_val_bsz1/ --data-filename DETR_Run100 --batch-size 1 --threshold 0.2 --tolerance 5 --init-num 30 --check-npz-path ../Results/stat_npzs/P100_DETR_PyTorch_val_bsz1.npz  --fit-type kde --n-level specific --detect-type cumulate --dataset-type COCO --combine-type i --rm-outs-type none --step 5 --quality-type map --check-min-n

## DETR
# python calculate_stats.py --data-dir ../Results/Raw/P100_DETR_TensorFlow_val_bsz2/ --data-filename DETR_Run100 --batch-size 2 --threshold 0.05 --tolerance 5 --init-num 30 --check-npz-path ../Results/stat_npzs/P100_DETR_TensorFlow_val_bsz2.npz  --fit-type kde --n-level specific --detect-type cumulate --dataset-type COCO --combine-type i --rm-outs-type none --step 5 --quality-type map --check-min-n

python calculate_stats.py --data-dir ../Results/Raw/P100_DETR_PyTorch_val_bsz2/ --data-filename DETR_Run100 --batch-size 2 --threshold 0.2 --tolerance 5 --init-num 30 --check-npz-path ../Results/stat_npzs/P100_DETR_PyTorch_val_bsz2.npz  --fit-type kde --n-level specific --detect-type cumulate --dataset-type COCO --combine-type i --rm-outs-type none --step 5 --quality-type map --check-min-n

## ViT
python calculate_stats.py --data-dir ../Results/Raw/P100_ViT_TensorFlow_val_bsz256/ --data-filename ViT_Run100 --batch-size 256 --threshold 0.2 --tolerance 5 --init-num 30 --check-npz-path ../Results/stat_npzs/P100_ViT_TensorFlow_val_bsz256.npz  --fit-type kde --n-level specific --detect-type cumulate --dataset-type ImageNet --combine-type i --rm-outs-type none --step 5 --quality-type acc --check-min-n

python calculate_stats.py --data-dir ../Results/Raw/P100_ViT_PyTorch_val_bsz256/ --data-filename ViT_Run100 --batch-size 256 --threshold 0.2 --tolerance 5 --init-num 30 --check-npz-path ../Results/stat_npzs/P100_ViT_PyTorch_val_bsz256.npz  --fit-type kde --n-level specific --detect-type cumulate --dataset-type ImageNet --combine-type i --rm-outs-type none --step 5 --quality-type acc --check-min-n