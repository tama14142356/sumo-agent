#!/bin/bash

set -Ceu

outdir_sub=""
if [ "$#" -gt 1 ]; then
        outdir_sub_dirname=$(dirname "$2")
        if [ "$outdir_sub_dirname" = "." ]; then
                outdir_sub_dirname=""
        else
                outdir_sub_dirname="$outdir_sub_dirname"/
        fi
        outdir_sub="$outdir_sub_dirname"$(basename "$2")/
elif [ "$#" -ne 1 ]; then
        echo "指定された引数は$#個です。" 1>&2
        echo "GPU番号を指定してください。" 1>&2
        exit 1
fi

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/sumo-light/pfrl_result/dqn/"$outdir_sub"$(basename "$tempdir")

mkdir -p "$outdir"/gym-sumo-version
pushd "$project_root"/../gym-sumo
git status > "$outdir"/gym-sumo-version/git-status.txt
git diff > "$outdir"/gym-sumo-version/git-diff.txt
git log > "$outdir"/gym-sumo-version/git-log.txt
git rev-parse HEAD > "$outdir"/gym-sumo-version/git-head.txt
popd

gpu="$1"

dir_name=$(dirname "$0")
cd "$dir_name"/../src

# gym/dqn
CUDA_VISIBLE_DEVICES="$gpu" python train_dqn_gym.py --outdir "$outdir"/train --gpu 0
model=$(find "$outdir"/train -name "best")
CUDA_VISIBLE_DEVICES="$gpu" python train_dqn_gym.py --eval-n-runs 1 --demo --monitor --load "$model" --outdir "$outdir"/demo/best --gpu 0
model=$(find "$outdir"/train -name "*_finish")
CUDA_VISIBLE_DEVICES="$gpu" python train_dqn_gym.py --eval-n-runs 1 --demo --monitor --load "$model" --outdir "$outdir"/demo/finish --gpu 0
