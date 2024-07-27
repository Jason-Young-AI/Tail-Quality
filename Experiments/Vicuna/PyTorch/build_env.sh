#!/bin/bash
source ./vars.sh
conda create -n ${THIS_ENV_NAME} python=3.10 -y
source $CONDA_PREFIX/bin/activate ${THIS_ENV_NAME}
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers --no-input
pip install tensor_parallel pandas scikit-learn matplotlib sentencepiece accelerate matplotlib --no-input