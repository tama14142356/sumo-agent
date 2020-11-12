#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/$(basename $tempdir)

gpu="$1"

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/categorical_dqn
CUDA_VISIBLE_DEVICES=$gpu python train_categorical_dqn_gym.py --outdir $outdir/categorical_dqn/train --gpu 0
model=$(find $outdir/categorical_dqn/train -name "*_finish")
CUDA_VISIBLE_DEVICES=$gpu python train_categorical_dqn_gym.py --demo --monitor --load $model --outdir $outdir/categorical_dqn/demo --gpu 0
