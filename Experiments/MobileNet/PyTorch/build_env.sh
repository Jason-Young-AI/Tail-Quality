#!/bin/bash
source ./vars.sh
conda create -n ${THIS_ENV_NAME} python=3.10 -y
source $CONDA_PREFIX/bin/activate ${THIS_ENV_NAME}
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
#conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch -y
pip install tqdm logging --no-input
pip install numpy matplotlib scikit-learn scipy tqdm KDEpy --no-input