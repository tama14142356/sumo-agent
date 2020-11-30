#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/sumo-light/pfrl_result/dqn/$(basename $tempdir)

gpu="$1"

dir_name=$(dirname $0)
cd $dir_name/../src

# gym/dqn
CUDA_VISIBLE_DEVICES=$gpu python train_dqn_gym.py --outdir $outdir/train --gpu 0
model=$(find $outdir/train -name "best")
CUDA_VISIBLE_DEVICES=$gpu python train_dqn_gym.py --eval-n-runs 1 --demo --monitor --load $model --outdir $outdir/best_demo --gpu 0
model=$(find $outdir/train -name "*_finish")
CUDA_VISIBLE_DEVICES=$gpu python train_dqn_gym.py --eval-n-runs 1 --demo --monitor --load $model --outdir $outdir/finish_demo --gpu 0
