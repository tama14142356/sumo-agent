#!/bin/bash
# install torch, torch-geometric
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
pip install torch-geometric

# other requirement packages
pip install -r requirements.txt
