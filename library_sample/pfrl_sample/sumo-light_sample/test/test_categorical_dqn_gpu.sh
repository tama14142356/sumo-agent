#!/bin/bash

set -Ceu

dir_name=$(dirname "$0")
pushd "$dir_name"
if [ "$#" -gt 1 ]; then
        bash test_gpu.sh "$1" train_categorical_dqn_gym.py categorical_dqn "$2"
else
        bash test_gpu.sh "$1" train_categorical_dqn_gym.py categorical_dqn
fi
popd
