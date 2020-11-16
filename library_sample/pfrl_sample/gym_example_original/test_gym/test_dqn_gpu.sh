#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/pfrl_result/dqn/$(basename $tempdir)

gpu="$1"

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/dqn
CUDA_VISIBLE_DEVICES=$gpu python train_dqn_gym.py --outdir $outdir/train --gpu 0
model=$(find $outdir/train -name "*_finish")
CUDA_VISIBLE_DEVICES=$gpu python train_dqn_gym.py --demo --monitor --load $model --outdir $outdir/demo --gpu 0
