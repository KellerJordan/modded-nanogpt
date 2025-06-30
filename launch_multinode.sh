#!/bin/bash

# Multi-node Multi-GPU Training Launch Script for SpeedRunningESM2
# This script sets up the environment and launches distributed training

# === CONFIGURATION ===
# Modify these variables according to your setup

# Number of nodes (machines)
NUM_NODES=${NUM_NODES:-1}

# Number of GPUs per node
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}

# Master node address (IP or hostname of rank 0 node)
MASTER_ADDR=${MASTER_ADDR:-localhost}

# Master port (choose an available port)
MASTER_PORT=${MASTER_PORT:-29500}

# Current node rank (0 for master, 1+ for workers)
NODE_RANK=${NODE_RANK:-0}

# Optional: NCCL settings for better performance
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}  # 2 hours
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}

# CUDA settings
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.0;7.5;8.0;8.6;8.9;9.0"}

# Python path (optional, uncomment if needed)
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# === LAUNCH COMMAND ===
echo "Launching distributed training..."
echo "Nodes: $NUM_NODES, GPUs per node: $GPUS_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT, Current node rank: $NODE_RANK"
echo "Total GPUs: $((NUM_NODES * GPUS_PER_NODE))"

# For single-node multi-GPU training
if [ $NUM_NODES -eq 1 ]; then
    echo "Running single-node multi-GPU training..."
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$GPUS_PER_NODE \
        train.py \
        --ddp_static_graph \
        --ddp_timeout 7200 \
        --clear_cache_every 1000 \
        --grad_clip 1.0 \
        "$@"
else
    # For multi-node training
    echo "Running multi-node training..."
    torchrun \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=speedrun_esm2 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --node_rank=$NODE_RANK \
        train.py \
        --ddp_static_graph \
        --ddp_timeout 7200 \
        --clear_cache_every 1000 \
        --grad_clip 1.0 \
        "$@"
fi

# === USAGE EXAMPLES ===
# Single-node, 8 GPUs:
# ./launch_multinode.sh --wandb_token YOUR_TOKEN --save_path model_name

# Multi-node (on master node with rank 0):
# NUM_NODES=2 MASTER_ADDR=10.0.0.1 NODE_RANK=0 ./launch_multinode.sh --wandb_token YOUR_TOKEN

# Multi-node (on worker node with rank 1):
# NUM_NODES=2 MASTER_ADDR=10.0.0.1 NODE_RANK=1 ./launch_multinode.sh --wandb_token YOUR_TOKEN

# Custom settings:
# GPUS_PER_NODE=4 NCCL_DEBUG=WARN ./launch_multinode.sh --batch_size 32768 --grad_accum 2 