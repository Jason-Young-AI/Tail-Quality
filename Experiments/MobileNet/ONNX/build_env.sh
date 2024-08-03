#!/bin/bash
source ./vars.sh
conda create -n ${THIS_ENV_NAME} python=3.8 -y
source $CONDA_PREFIX/bin/activate ${THIS_ENV_NAME}
pip install future --no-input
conda install cudatoolkit=10.2 -c pytorch -y
pip install onnx requests==2.20 --no-input
pip install mxnet-cu102mkl --pre -U --no-input
pip install tqdm --no-input
pip install numpy==1.21 pandas==1.3.5 matplotlib==3.3.4 scikit-learn scipy importlib-metadata KDEpy --no-input
pip install gluoncv --no-input
