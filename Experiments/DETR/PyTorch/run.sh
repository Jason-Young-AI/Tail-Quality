WORKING_DIR=$(cd "$(dirname "$0")";pwd)
echo $WORKING_DIR

cd $WORKING_DIR/sources/with_dilation_aka_bsz1
pwd
mkdir -p ../../results/bsz1/
python run.py --run-number 100 --batch-size 1 --dataset-path ~/1.Datasets/MS-COCO --results-basepath ../../results/bsz1/DETR_Run100 --local --model-path ../../bins/DETR_ResNet101_BSZ1.pth

cd $WORKING_DIR/sources/without_dilation_aka_bsz2
pwd
mkdir -p ../../results/bsz2/
python run.py --run-number 100 --batch-size 2 --dataset-path ~/1.Datasets/MS-COCO --results-basepath ../../results/bsz2/DETR_Run100 --local --model-path ../../bins/DETR_ResNet101_BSZ2.pth
