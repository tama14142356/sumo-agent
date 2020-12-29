#!/bin/bash
# install torch, torch-geometric
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install torch-geometric

# other requirement packages
pip install -r requirements.txt
