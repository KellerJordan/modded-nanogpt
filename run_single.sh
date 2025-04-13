#!/bin/bash

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,3 torchrun --standalone --nproc_per_node=2 \
    train_gpt2_main.py \
    --data_path "data/shakespeare_char" \
    --device_batch_size 16 \
    --batch_size 32 \
    --num_iterations 300 \
    --generate_every 100 \
    --n_heads 6 \
    --n_layers 6 \
    --head_dim 16 \
    --sequence_length 256 \
    --attn_mode "euc" \
    --head_mode "euc" \
    --curvature 1. \
    --k_lr 0. \
    --seed 0 \
    > logs/last2.txt 2>&1
    
# --curvature "${curvature}" \
# --k_lr "${k_lr}" \
# --seed "${seed}" \