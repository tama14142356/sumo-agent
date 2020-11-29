#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/atari/pfrl_result/ppo/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../../src

# atari/ppo
python train_ppo_ale.py --steps 100 --update-interval 50 --batchsize 16 --epochs 2 --outdir $outdir/train --gpu -1
model=$(find $outdir/train -name "*_finish")
python train_ppo_ale.py --demo --load $model --eval-n-runs 1 --outdir $outdir/demo_finish --gpu -1
# model=$(find $outdir/train -name "best")
# python train_ppo_ale.py --monitor --demo --load $model --eval-n-runs 2 --outdir $outdir/demo_best --gpu -1
