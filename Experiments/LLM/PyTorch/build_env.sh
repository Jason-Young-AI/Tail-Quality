#!/bin/bash
conda create -n TQ_LLM_PyTorch python cython scipy pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y
source ${CONDA_PREFIX}/bin/activate TQ_LLM_PyTorch
echo ${CONDA_PREFIX}
pip install --upgrade pip
pip install transformers
pip install tensor_parallel pandas scikit-learn matplotlib sentencepiece accelerate
pip install YoungToolkit pycocotools pandas accelerate sentencepiece
