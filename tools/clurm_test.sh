#!/bin/bash
###
 # @Author: AristoYU
 # @Date: 2024-02-17 10:56:41
 # @LastEditTime: 2024-02-17 10:56:43
 # @LastEditors: AristoYU
 # @Description: 
 # @FilePath: /mm_perception/tools/clurm_test.sh
### 

set -x

PARTITION=$1
JOB_NAME=$2
TASK=$3
CONFIG=$4
CHECKPOINT=$5
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:6}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}