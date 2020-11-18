#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/sumo-light/pfrl_result/dqn/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/dqn
python train_dqn_gym.py --steps 10000 --replay-start-size 50 --outdir $outdir/train --gpu -1
model=$(find $outdir/train -name "best")
python train_dqn_gym.py --eval-n-runs 1 --demo --monitor --load $model --eval-n-runs 100 --outdir $outdir/best_demo --gpu -1
model=$(find $outdir/train -name "*_finish")
python train_dqn_gym.py --eval-n-runs 1 --demo --monitor --load $model --eval-n-runs 100 --outdir $outdir/finish_demo --gpu -1
