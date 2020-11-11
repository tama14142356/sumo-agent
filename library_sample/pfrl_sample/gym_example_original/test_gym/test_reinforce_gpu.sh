#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/reinforce
CUDA_VISIBLE_DEVICES=$gpu python train_reinforce_gym.py --steps 100 --batchsize 1 --outdir $outdir/gym/reinforce --gpu 0
model=$(find $outdir/gym/reinforce -name "*_finish")
CUDA_VISIBLE_DEVICES=$gpu python train_reinforce_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu 0
