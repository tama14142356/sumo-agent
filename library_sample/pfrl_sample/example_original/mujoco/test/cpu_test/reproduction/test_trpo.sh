#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/mujoco/pfrl_result/trpo/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../../src/reproduction/trpo

# mujoco/reproduction/trpo (specify non-mujoco env to test without mujoco)
python train_trpo.py --steps 10 --trpo-update-interval 5 --outdir $outdir/train --env Pendulum-v0 --gpu -1
model=$(find $outdir/train -name "*_finish")
python train_trpo.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/demo --gpu -1
