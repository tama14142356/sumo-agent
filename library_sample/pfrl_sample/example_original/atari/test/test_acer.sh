#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/atari/pfrl_result/acer/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../src

gpu="$1"

# atari/acer (only for cpu)
if [[ $gpu -lt 0 ]]; then
  python train_acer_ale.py 4 --env PongNoFrameskip-v4 --steps 100 --outdir $outdir/train
  model=$(find $outdir/train -name "*_finish")
  python train_acer_ale.py 4 --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/demo
fi
