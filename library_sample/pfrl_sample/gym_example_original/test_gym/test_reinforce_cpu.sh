#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/reinforce
python train_reinforce_gym.py --steps 10000 --batchsize 128 --outdir $outdir/gym/reinforce --gpu -1
model=$(find $outdir/gym/reinforce -name "*_finish")
python train_reinforce_gym.py --demo --monitor --load $model --eval-n-runs 100 --outdir $outdir/temp --gpu -1

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
mv $outdir/ $project_root/results/
echo Output files are moved to $project_root/results/$(basename $outdir)
