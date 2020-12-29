#!/bin/bash

set -Ceu

outdir_sub=""
alg_name=""
if [ "$#" -gt 3 ]; then
        outdir_sub_dirname=$(dirname "$4")
        if [ "$outdir_sub_dirname" = "." ]; then
                outdir_sub_dirname=""
        else
                outdir_sub_dirname="$outdir_sub_dirname"/
        fi
        outdir_sub="$outdir_sub_dirname"$(basename "$4")/
fi
if [ "$#" -gt 2 ]; then
        alg_name_dirname=$(dirname "$3")
        if [ "$alg_name_dirname" = "." ]; then
                alg_name_dirname=""
        else
                alg_name_dirname="$alg_name_dirname"/
        fi
        alg_name="$alg_name_dirname"$(basename "$3")/
fi
if [ "$#" -gt 1 ]; then
        python_file_name=$(basename "$2")
else
        echo "指定された引数は$#個です。" 1>&2
        echo "GPU番号、ファイル名を指定してください。" 1>&2
        exit 1
fi

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
tempdir=$(mktemp -d)
outdir=$project_root/results/sumo-light/pfrl_result/"$alg_name""$outdir_sub"$(basename "$tempdir")

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

CUDA_VISIBLE_DEVICES="$gpu" python "$python_file_name" --outdir "$outdir"/train --gpu 0
model=$(find "$outdir"/train -name "*_finish")
if [ "$model" == "" ]; then
        echo "can't find finish model"
else
        CUDA_VISIBLE_DEVICES="$gpu" python "$python_file_name" --eval-n-runs 1 --demo --monitor --load "$model" --outdir "$outdir"/demo/finish --gpu 0
fi
model=$(find "$outdir"/train -name "best")
if [ "$model" == "" ]; then
        echo "can't find best model"
else
        CUDA_VISIBLE_DEVICES="$gpu" python "$python_file_name" --eval-n-runs 1 --demo --monitor --load "$model" --outdir "$outdir"/demo/best --gpu 0
fi
