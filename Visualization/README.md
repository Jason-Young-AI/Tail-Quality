python extract_data.py --data-dir ../results/raw/P100/ViT/PyTorch/val/bsz256/ -n ViT_Run100 -t ImageNet
python extract_data.py --data-dir ../results/raw/P100/DETR/PyTorch/val/bsz1/ -n DETR_Run100 -t COCO
python extract_data.py --data-dir ../results/raw/V100/LLM/PyTorch/val/bsz1/ -n LLM_Run100 -t MMLU

python calculate_quality.py --data-dir ../results/raw/P100/ViT/PyTorch/val/bsz256 --data-filename ViT_Run100  --run-index 22 --threshold 1 --assets-path ./assets --quality-type acc --dataset-type ImageNet  --combine-type i
python calculate_quality.py --data-dir ../results/raw/P100/DETR/PyTorch/val/bsz1/ --data-filename DETR_Run100  --run-index 22 --threshold 1 --assets-path ./assets --quality-type map --dataset-type COCO  --combine-type i
python calculate_quality.py --data-dir ../results/raw/V100/LLM/PyTorch/val/bsz1/ --data-filename LLM_Run100  --run-index 22 --threshold 1 --assets-path ./assets --quality-type acc --dataset-type MMLU  --combine-type i

python draw.py --save-dir ../results/figs/P100_ViT_PT_val_bsz256 --data-dir ../results/raw/P100/ViT/PyTorch/val/bsz256/ -n ViT_Run100 --dataset-type ImageNet -c i
python draw.py --save-dir ../results/figs/P100_DETR_PT_val_bsz1 --data-dir ../results/raw/P100/DETR/PyTorch/val/bsz1/ -n DETR_Run100 --dataset-type COCO -c i
python draw.py --save-dir ../results/figs/V100_LLM_PT_val_bsz1 --data-dir ../results/raw/V100/LLM/PyTorch/val/bsz1/ -n LLM_Run100 --dataset-type MMLU -c i

python draw_qualities.py --save-dir ../results/figs/P100_ViT_PT_val_bsz256/ --data-dir ../results/raw/P100/ViT/PyTorch/val/bsz256 --data-filename ViT_Run100 --sub-proc-num 8 --interplt-num 10 --quality-type acc --dataset-type ImageNet --combine-type i --rm-outs-type gaussian --recompute --draw-npz-path ../results/P100_ViT_PT_val_bsz256
python draw_qualities.py --save-dir ../results/figs/P100_DETR_PT_val_bsz1/ --data-dir ../results/raw/P100/DETR/PyTorch/val/bsz1 --data-filename DETR_Run100 --sub-proc-num 8 --interplt-num 10 --quality-type map --dataset-type COCO --combine-type i --rm-outs-type gaussian --recompute --draw-npz-path ../results/P100_DETR_PT_val_bsz1
python draw_qualities.py --save-dir ../results/figs/V100_LLM_PT_val_bsz1/ --data-dir ../results/raw/V100/LLM/PyTorch/val/bsz1 --data-filename LLM_Run100 --sub-proc-num 8 --interplt-num 10 --quality-type acc --dataset-type MMLU --combine-type i --rm-outs-type gaussian --recompute --draw-npz-path ../results/V100_LLM_PT_val_bsz1