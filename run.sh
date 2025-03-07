#!/bin/bash

SEEDS=(1 2 3 4)
NUM_ITERATIONS=5000  
GENERATE_EVERY=0
CURVATURES=(1.)
K_LRS=(10.)

run_experiment() {
    local exp_name="$1"
    local head_mode="$2"
    local attn_mode="$3"
    local dataset="$4" 
    local gpu_config="$5"
    local n_heads="$6"
    local n_layers="$7"
    local head_dim="$8"
    local device_batch="$9"

    for seed in "${SEEDS[@]}"; do
        local dataset_name
        dataset_name=$(basename "$dataset")

        for curvature in "${CURVATURES[@]}"; do
            for k_lr in "${K_LRS[@]}"; do
                echo "Running on ${dataset_name}"
                CUDA_VISIBLE_DEVICES="${gpu_config}" torchrun --standalone --nproc_per_node=1 \
                    train_gpt2_main.py \
                    --data_path "$dataset" \
                    --num_iterations "$NUM_ITERATIONS" \
                    --generate_every "$GENERATE_EVERY" \
                    --head_mode "$head_mode" \
                    --attn_mode "$attn_mode" \
                    --curvature "$curvature" \
                    --k_lr "$k_lr" \
                    --seed "$seed" \
                    --n_heads "$n_heads" \
                    --n_layers "$n_layers" \
                    --head_dim "$head_dim" \
                    --device_batch_size "$device_batch" \
                    --batch_size 512 \
                    > logs/${dataset_name}_seed${seed}.txt 2>&1 # "logs/${dataset_name}_${exp_name}_k${curvature}_lr${k_lr}_seed${seed}.txt" 2>&1
            done
        done
    done
}



# Run all experiments
run_experiment "fully_euc" "euc" "euc" "data/fineweb10B" 6 6 12 128 32 
run_experiment "hyp_head" "hyp" "euc" "data/fineweb10B" 6 6 12 128 32 


echo "All experiments completed!"