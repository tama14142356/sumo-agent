#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/categorical_dqn
CUDA_VISIBLE_DEVICES=$gpu python train_categorical_dqn_gym.py --steps 100 --replay-start-size 50 --outdir $outdir/gym/categorical_dqn --gpu 0
model=$(find $outdir/gym/categorical_dqn -name "*_finish")
CUDA_VISIBLE_DEVICES=$gpu python train_categorical_dqn_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu 0
