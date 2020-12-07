#!/bin/bash

set -Ceu

dir_name=$(dirname "$0")
pushd "$dir_name"
if [ "$#" -gt 1 ]; then
        bash test_gpu.sh "$1" train_ppo.py ppo "$2"
else
        bash test_gpu.sh "$1" train_ppo.py ppo
fi
popd
