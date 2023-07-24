WORKING_DIR=$(cd "$(dirname "$0")";pwd)
echo $WORKING_DIR

cd $WORKING_DIR/sources
pwd
mkdir -p ../results/bsz1/
python run.py --run-number 100 --batch-size 1 --dataset-path ~/1.Datasets/MS-COCO --results-basepath ../results/bsz1/DETR_Run100 --local --model-path ../bins/detr-r101-dc5-a2e86def.h5

mkdir -p ../results/bsz2/
python run.py --run-number 100 --batch-size 2 --dataset-path ~/1.Datasets/MS-COCO --results-basepath ../results/bsz2/DETR_Run100 --local --model-path ../bins/detr-r101-2c7b67e5.h5
