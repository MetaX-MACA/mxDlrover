# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

export MCCL_NET_GDR_LEVEL=7
export MCCL_MAX_NCHANNELS=16
export MCCL_P2P_LEVEL=SYS
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export FORCE_ACTIVATE_WAIT=1
export GLOO_SOCKET_IFNAME=ens5f0np0
export MCCL_IB_HCA=mlx5_0,mlx5_1
export DYNAMIC_QUEUE_SCHEDULE=1

export PYTHONPATH=/gpfs/userdata/m00915/colossalai_train/ColossalAI-master:/gpfs/userdata/m00915/dlrover_version_test/dlrover-new


NNODES=${WORLD_SIZE}
# NNODES=1
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_REAL_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=${RANK}

TP=1
PP=8
DP=$((${GPU_NUM}/${TP}/${PP}))
CKPT_PATH="/gpfs/userdata/m00915/colossalai_train/ckpt"
MASTER_ADDR="127.0.0.1"
BATCH_SIZE=1024

python -m dlrover.trainer.torch.elastic_run --nnodes=$NNODES --nproc_per_node=${GPUS_PER_NODE} \
    /gpfs/userdata/m00915/colossalai_train/colossalai_train/colossalai_model_test.py \
    --batch_size=${BATCH_SIZE} -p=3d_cpu --pp=${PP} --tp=${TP} \
    --dataset=random \
    --load_checkpoint=${CKPT_PATH} \
    --save_dir=${CKPT_PATH} 

# --master-addr=${MASTER_ADDR} --master-port=12333 \
# --load_checkpoint=${CKPT_PATH} \