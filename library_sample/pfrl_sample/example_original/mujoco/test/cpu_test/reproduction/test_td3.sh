#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/mujoco/pfrl_result/td3/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../../src/reproduction/td3

# mujoco/reproduction/td3 (specify non-mujoco env to test without mujoco)
python train_td3.py --env Pendulum-v0 --gpu -1 --steps 10 --replay-start-size 5 --batch-size 5 --outdir $outdir/train
model=$(find $outdir/train -name "*_finish")
python train_td3.py --env Pendulum-v0 --demo --load $model --eval-n-runs 1 --outdir $outdir/demo --gpu -1
