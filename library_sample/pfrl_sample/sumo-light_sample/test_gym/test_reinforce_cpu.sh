#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/reinforce
python train_reinforce_gym.py --steps 100 --batchsize 1 --outdir $outdir/gym/reinforce
model=$(find $outdir/gym/reinforce -name "*_finish")
python train_reinforce_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp
