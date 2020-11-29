#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/mujoco/pfrl_result/ddpg/$(basename $tempdir)

gpu="$1"

dir_name=$(dirname $0)
cd $dir_name/../../src/reproduction/ddpg

# mujoco/reproduction/ddpg (specify non-mujoco env to test without mujoco)
CUDA_VISIBLE_DEVICES=$gpu python train_ddpg.py --env Pendulum-v0 --gpu 0 --steps 10 --replay-start-size 5 --batch-size 5 --outdir $outdir/train
model=$(find $outdir/train -name "*_finish")
CUDA_VISIBLE_DEVICES=$gpu python train_ddpg.py --env Pendulum-v0 --demo --load $model --eval-n-runs 1 --outdir $outdir/demo --gpu 0
