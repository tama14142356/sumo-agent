#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/dqn
CUDA_VISIBLE_DEVICES=$gpu python train_dqn_gym.py --steps 100 --replay-start-size 50 --outdir $outdir/gym/dqn --gpu 0
model=$(find $outdir/gym/dqn -name "*_finish")
CUDA_VISIBLE_DEVICES=$gpu python train_dqn_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu 0
