#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/atari/pfrl_result/a2c/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../../src

# atari/a2c
python train_a2c_ale.py --env PongNoFrameskip-v4 --steps 100 --update-steps 50 --outdir $outdir/train --gpu -1
model=$(find $outdir/train -name "*_finish")
python train_a2c_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/demo --gpu -1
