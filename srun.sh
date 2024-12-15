#!/bin/bash

# Sequence of curvature values
curvatures=(10.0 15.0 20.0 25.0 30.0 40.0 50.0)

# Iterate over each curvature value
for curvature in "${curvatures[@]}"; do
    echo "Running script with curvature=${curvature}"
    CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 train_gpt2_hyp.py --curvature "${curvature}" > logs_curvature_${curvature}.txt 2>&1
    echo "Finished run with curvature=${curvature}, logs saved to logs_curvature_${curvature}.txt"
done
