#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/mujoco/pfrl_result/soft_actor_critic/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../../src/reproduction/soft_actor_critic

# mujoco/soft_actor_critic (specify non-mujoco env to test without mujoco)
python train_soft_actor_critic.py --env Pendulum-v0 --gpu -1 --steps 10 --replay-start-size 5 --batch-size 5 --outdir $outdir/train
model=$(find $outdir/train -name "*_finish")
python train_soft_actor_critic.py --env Pendulum-v0 --demo --load $model --eval-n-runs 1 --outdir $outdir/demo --gpu -1
