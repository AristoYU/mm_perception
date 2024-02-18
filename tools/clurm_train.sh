#!/bin/bash
###
 # @Author: AristoYU
 # @Date: 2024-02-17 10:51:09
 # @LastEditTime: 2024-02-17 10:51:12
 # @LastEditors: AristoYU
 # @Description: 
 # @FilePath: /mm_perception/tools/clurm_train.sh
### 

set -x

PARTITION=$1
JOB_NAME=$2
TASK=$3
CONFIG=$4
WORK_DIR=$5
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:6}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train_$TASK.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}