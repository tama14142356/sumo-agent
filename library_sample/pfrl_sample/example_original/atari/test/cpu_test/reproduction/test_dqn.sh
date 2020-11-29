#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/atari/pfrl_result/reproduction/dqn/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../../../src/reproduction/dqn

# atari/reproduction/dqn
python train_dqn.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/train --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1 --gpu -1
model=$(find $outdir/train -name "*_finish")
python train_dqn.py --env PongNoFrameskip-v4 --demo --load $model --outdir $outdir/demo --eval-n-steps 200 --gpu -1
