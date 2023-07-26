WORKING_DIR=$(cd "$(dirname "$0")";pwd)
echo $WORKING_DIR

cd $WORKING_DIR/sources
python imagenet.py --save-dir ../dataset --dataset-path ~/1.Datasets/ImageNet2012/
