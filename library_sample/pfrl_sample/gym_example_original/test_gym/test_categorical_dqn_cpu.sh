#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/categorical_dqn
python train_categorical_dqn_gym.py --steps 10000 --replay-start-size 50 --outdir $outdir/categorical_dqn/train --gpu -1
model=$(find $outdir/categorical_dqn/train -name "*_finish")
python train_categorical_dqn_gym.py --demo --monitor --load $model --eval-n-runs 100 --outdir $outdir/categorical_dqn/demo --gpu -1
