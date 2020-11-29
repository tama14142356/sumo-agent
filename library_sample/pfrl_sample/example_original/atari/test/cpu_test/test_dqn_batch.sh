#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/atari/pfrl_result/dqn_batch/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../../src

# atari/dqn_batch
python train_dqn_batch_ale.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/train --gpu -1
model=$(find $outdir/train -name "*_finish")
python train_dqn_batch_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/demo --gpu -1
