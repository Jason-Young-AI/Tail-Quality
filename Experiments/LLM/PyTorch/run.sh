WORKING_DIR=$(cd "$(dirname "$0")";pwd)
echo $WORKING_DIR

cd $WORKING_DIR/sources
pwd
mkdir -p ../results/val/bsz1/
python run.py --run-number 100 --batch-size 1 --dataset-path ../data --results-basepath ../results/val/bsz1/LLM_Run100 --local --model-path lmsys/vicuna-13b-v1.3 --run-mode val

#mkdir -p ../results/test/bsz1/
#python run.py --run-number 100 --batch-size 1 --dataset-path ../data --results-basepath ../results/test/bsz1/LLM_Run100 --local --model-path lmsys/vicuna-13b-v1.3 --run-mode test