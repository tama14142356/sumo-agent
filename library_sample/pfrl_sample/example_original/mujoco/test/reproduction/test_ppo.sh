#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/mujoco/pfrl_result/ppo/$(basename $tempdir)

gpu="$1"

dir_name=$(dirname $0)
cd $dir_name/../../src/reproduction/ppo

# mujoco/reproduction/ppo (specify non-mujoco env to test without mujoco)
CUDA_VISIBLE_DEVICES=$gpu python train_ppo.py --steps 10 --update-interval 5 --batch-size 5 --epochs 2 --outdir $outdir/train --env Pendulum-v0 --gpu 0
model=$(find $outdir/train -name "*_finish")
CUDA_VISIBLE_DEVICES=$gpu python train_ppo.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/demo --gpu 0
