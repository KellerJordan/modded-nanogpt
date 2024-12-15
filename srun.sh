#!/bin/bash

# Sequence of curvature values
curvatures=(100.0 200.0 500.0 1000.0)

# Sequence of seeds
seeds=(1 2 3 4)

# First, run Euclidean training with all seeds
for seed in "${seeds[@]}"; do
    echo "Running Euclidean model training with seed=${seed}"
    CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 train_gpt2_hyp.py --lm_head 'euc' --seed "${seed}" > logs/logs_euclidean_seed_${seed}.txt 2>&1
    echo "Finished Euclidean run with seed=${seed}, logs saved to logs/logs_euclidean_seed_${seed}.txt"
done

# Now, run training for each curvature and each seed
for curvature in "${curvatures[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Running script with curvature=${curvature} and seed=${seed}"
        CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 train_gpt2_hyp.py --lm_head 'hyp' --curvature "${curvature}" --seed "${seed}" > logs/logs_curvature_${curvature}_seed_${seed}.txt 2>&1
        echo "Finished run with curvature=${curvature} and seed=${seed}, logs saved to logs/logs_curvature_${curvature}_seed_${seed}.txt"
    done
done

echo "All experiments completed!"
