WORKING_DIR=$(cd "$(dirname "$0")";pwd)
echo $WORKING_DIR

cd $WORKING_DIR/sources
pwd

mkdir -p ../results/bsz256/
python run.py --run-number 100 --batch-size 256 --dataset-path ../dataset --results-basepath ../results/bsz256/ViT_Run100 --local --model-path ../bins/jx_vit_large_p32_384-9b920ba8.pth

mkdir -p ../results/bsz512/
python run.py --run-number 100 --batch-size 512 --dataset-path ../dataset --results-basepath ../results/bsz512/ViT_Run100 --local --model-path ../bins/jx_vit_large_p32_384-9b920ba8.pth

mkdir -p ../results/bsz1024/
python run.py --run-number 100 --batch-size 1024 --dataset-path ../dataset --results-basepath ../results/bsz1024/ViT_Run100 --local --model-path ../bins/jx_vit_large_p32_384-9b920ba8.pth

