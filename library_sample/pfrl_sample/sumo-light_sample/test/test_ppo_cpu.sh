#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/sumo-light/pfrl_result/ppo/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../src

# mujoco/reproduction/ppo (specify non-mujoco env to test without mujoco)
python train_ppo.py --outdir $outdir/train --gpu -1
# model=$(find $outdir/train -name "best")
# python train_ppo.py --demo --load $model --eval-n-runs 1 --monitor --outdir $outdir/demo/best --gpu -1
model=$(find $outdir/train -name "*_finish")
python train_ppo.py --demo --load $model --eval-n-runs 1 --monitor --outdir $outdir/demo/finish --gpu -1
