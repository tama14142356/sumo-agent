#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/atari/pfrl_result/reproduction/iqn/$(basename $tempdir)

gpu="$1"

dir_name=$(dirname $0)
cd $dir_name/../../src/reproduction/iqn

# atari/reproduction/iqn
CUDA_VISIBLE_DEVICES=$gpu python train_iqn.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/train --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1  --gpu 0
model=$(find $outdir/train -name "*_finish")
CUDA_VISIBLE_DEVICES=$gpu python train_iqn.py --env PongNoFrameskip-v4 --demo --load $model --outdir $outdir/demo --eval-n-steps 200 --gpu 0
