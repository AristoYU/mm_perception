#!/bin/bash
###
 # @Author: AristoYU
 # @Date: 2024-02-17 10:49:40
 # @LastEditTime: 2024-02-17 10:50:45
 # @LastEditors: AristoYU
 # @Description: 
 # @FilePath: /mm_perception/tools/dist_train.sh
### 

TASK=$1
CONFIG=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_$TASK.py \
    $CONFIG \
    --launcher pytorch ${@:4}
