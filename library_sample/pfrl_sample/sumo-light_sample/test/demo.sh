#!/bin/bash

set -Ceu

if [ "$#" -lt 3 ]; then
        echo "指定された引数は$#個です。" 1>&2
        echo "GPU番号、ファイル名、モデルのディレクトリ名、デモのサブディレクトリ名を指定してください。" 1>&2
        exit 1
fi
demo_sub_dir=""
if [ "$#" -gt 4 ]; then
        demo_sub_dir="$5"
fi
eval_n_runs=1
if [ "$#" -gt 3 ]; then
        eval_n_runs="$4"
fi

python_file_name=$(basename "$2")
model="$3"
model=$(cd "$model"; pwd)
model_dirname=$(dirname "$model")
while :
        if [ "$(basename "$model_dirname")" == "train" ]; then
                break
        fi
        do model_dirname=$(dirname "$model_dirname")
done
model_dirname=$(dirname "$model_dirname")
model_name=$(basename "$model")
outdir="$model_dirname"/demo/"$model_name"/not_auto/"$demo_sub_dir"

project_root=$(git rev-parse --show-superproject-working-tree --show-toplevel | head -1)
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

if [ "$gpu" -lt 0 ]; then
        python "$python_file_name" --eval-n-runs "$eval_n_runs" --demo --monitor --load "$model" --outdir "$outdir" --gpu "$gpu"
else
        CUDA_VISIBLE_DEVICES="$gpu" python "$python_file_name" --eval-n-runs "$eval_n_runs" --demo --monitor --load "$model" --outdir "$outdir" --gpu 0
fi
