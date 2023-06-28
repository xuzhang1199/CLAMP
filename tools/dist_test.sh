#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
function rand(){
    min=$1
    max=$(($2-$min+1))

    num=$(date +%s%N)
    echo $(($num%$max+$min))
}
rnd=$(rand 29500 30500)

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
GPU_ID=$4
PORT=${PORT:-$rnd}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPU_ID \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:5}