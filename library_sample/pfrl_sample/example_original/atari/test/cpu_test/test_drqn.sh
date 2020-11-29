#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/atari/pfrl_result/drqn/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../../src

# atari/drqn
python train_drqn_ale.py --env PongNoFrameskip-v4 --replay-start-size 50 --outdir $outdirt/train --gpu -1 --recurrent --flicker
model=$(find $outdirt/train -name "*_finish")
python train_drqn_ale.py --env PongNoFrameskip-v4 --demo --load $model --demo-n-episodes 1 --max-frames 50 --outdir $outdir/demo --gpu -1 --recurrent --flicker
