#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/categorical_dqn
CUDA_VISIBLE_DEVICES=$gpu python train_categorical_dqn_gym.py --outdir $outdir/gym/categorical_dqn --gpu 0
model=$(find $outdir/gym/categorical_dqn -name "*_finish")
CUDA_VISIBLE_DEVICES=$gpu python train_categorical_dqn_gym.py --demo --monitor --load $model --outdir $outdir/temp --gpu 0

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
mv $outdir/ $project_root/results/
echo Output files are moved to $project_root/results/$(basename $outdir)
