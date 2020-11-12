#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/reinforce
python train_reinforce_gym.py --steps 10000 --batchsize 128 --outdir $outdir/reinforce/train --gpu -1
model=$(find $outdir/reinforce/train -name "*_finish")
python train_reinforce_gym.py --demo --monitor --load $model --eval-n-runs 100 --outdir $outdir/reinforce/demo --gpu -1
