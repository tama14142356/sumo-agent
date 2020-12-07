#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "指定された引数は$#個です。" 1>&2
    echo "GPU番号を一つ以上指定してください。" 1>&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=$1 "${@:2}"
