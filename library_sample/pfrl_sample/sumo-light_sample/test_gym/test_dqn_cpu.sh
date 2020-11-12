#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

dir_name=$(dirname $0)
cd $dir_name/../gym

# gym/dqn
python train_dqn_gym.py --steps 10000 --replay-start-size 50 --outdir $outdir/gym/dqn --gpu -1
model=$(find $outdir/gym/dqn -name "*_finish")
python train_dqn_gym.py --demo --monitor --load $model --eval-n-runs 100 --outdir $outdir/temp --gpu -1

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
mv $outdir/ $project_root/results/
echo Output files are moved to $project_root/results/$(basename $outdir)
