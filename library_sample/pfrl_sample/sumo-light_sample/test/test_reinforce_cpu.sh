#!/bin/bash

set -Ceu

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/sumo-light/pfrl_result/reinforce/$(basename $tempdir)

dir_name=$(dirname $0)
cd $dir_name/../src

# gym/reinforce
python train_reinforce_gym.py --steps 10000 --batchsize 128 --outdir $outdir/train --gpu -1
model=$(find $outdir/train -name "best")
python train_reinforce_gym.py --eval-n-runs 1 --demo --monitor --load $model --outdir $outdir/demo/best --gpu -1
model=$(find $outdir/train -name "*_finish")
python train_reinforce_gym.py --eval-n-runs 1 --demo --monitor --load $model --outdir $outdir/demo/finish --gpu -1
