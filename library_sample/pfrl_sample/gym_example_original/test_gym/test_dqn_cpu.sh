#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/dqn
python train_dqn_gym.py --steps 100 --replay-start-size 50 --outdir $outdir/gym/dqn --gpu -1
model=$(find $outdir/gym/dqn -name "*_finish")
python train_dqn_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu -1
