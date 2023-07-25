conda create -n TQ_DETR_PyTorch python cython scipy pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y
source $CONDA_PREFIX/bin/activate TQ_DETR_PyTorch
pip install YoungToolkit pycocotools
