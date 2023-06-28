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
GPUS=$2
GPU_ID=$3
#PORT=${PORT:-29500}
PORT=${PORT:-$rnd}
echo $PORT
#echo "$(dirname $0)/.."
#echo $PYTHONPATH
#echo "$(dirname $0)/..":$PYTHONPATH

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPU_ID \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4}