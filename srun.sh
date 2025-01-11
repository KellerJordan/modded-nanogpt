#!/bin/bash

curvatures=(300. 1000. 3000.)
rates=(0.1 1.)
seeds=(1)

# for seed in "${seeds[@]}"; do
#     echo "Running script with euclidean head and seed=${seed}"
#     CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
#         train_gpt2_hyp.py \
#         --lm_head 'euc' \
#         --seed "${seed}" \
#         > logs/fineweb_euc_seed_${seed}.txt 2>&1
#     echo "Finished run with euclidean head and seed=${seed}, logs saved to logs/fineweb_euc_seed_${seed}.txt"
# done

for curvature in "${curvatures[@]}"; do
    # learnable
    for seed in "${seeds[@]}"; do
        for lr in "${rates[@]}"; do
            echo "Running script with learnable (lr=${lr}) curvature from ${curvature} and seed=${seed}"
            CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
                train_gpt2_hyp.py \
                --lm_head 'hyp' \
                --learnable True \
                --k_lr "${lr}" \
                --curvature "${curvature}" \
                --seed "${seed}" \
                > logs/fineweb_learnable_${lr}_curvature_${curvature}_seed_${seed}.txt 2>&1
            echo "Finished run with learnable (lr=${lr}) curvature=${curvature} and seed=${seed}, logs saved to logs/fineweb_learnable_${lr}_curvature_${curvature}_seed_${seed}.txt"
        done
        # not learnable
        for seed in "${seeds[@]}"; do
            echo "Running script with fixed curvature=${curvature} and seed=${seed}"
            CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
                train_gpt2_hyp.py \
                --lm_head 'hyp' \
                --curvature "${curvature}" \
                --seed "${seed}" \
                > logs/fineweb_curvature_${curvature}_seed_${seed}.txt 2>&1
            echo "Finished run with curvature=${curvature} and seed=${seed}, logs saved to logs/fineweb_curvature_${curvature}_seed_${seed}.txt"
        done
    done
done

echo "All experiments completed!"
