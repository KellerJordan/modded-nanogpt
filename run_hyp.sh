#!/bin/bash

# Dataset configurations
declare -A MODEL_CONFIGS=(
    ["data/tinystories_char"]="--n_heads 4 --head_dim 16 --n_layers 6 --device_batch_size 256 --batch_size 512"
    ["data/tinystories"]="--n_heads 6 --head_dim 32 --n_layers 6 --device_batch_size 32 --batch_size 64"
    ["data/fineweb10B"]="--n_heads 6 --head_dim 128 --n_layers 12 --device_batch_size 16 --batch_size 64"
)

# Experiment configurations
SEEDS=(0)
NUM_ITERATIONS=100  # Reduced for testing
DATASETS=("data/tinystories_char")
CURVATURES=(1.)
K_LRS=(10.)

run_experiment() {
    local exp_name=$1
    local head_mode=$2
    local attn_mode=$3
    local gpu_config=$4
    local extra_args=$5

    for seed in "${SEEDS[@]}"; do
        for path in "${DATASETS[@]}"; do
            dataset_name=$(basename ${path})
            model_config=${MODEL_CONFIGS[${path}]}

            if [ -n "$extra_args" ]; then
                # For hyperbolic experiments
                for curvature in "${CURVATURES[@]}"; do
                    for k_lr in "${K_LRS[@]}"; do
                        echo "Running ${exp_name} with seed=${seed}, k_lr=${k_lr}, k=${curvature} on ${dataset_name}"
                        CUDA_VISIBLE_DEVICES=${gpu_config} torchrun --standalone --nproc_per_node=2 \
                        train_gpt2_main.py \
                        --data_path "${path}" \
                        --num_iterations "${NUM_ITERATIONS}" \
                        --head_mode "${head_mode}" \
                        --attn_mode "${attn_mode}" \
                        --curvature "${curvature}" \
                        --k_lr "${k_lr}" \
                        --seed "${seed}" \
                        ${model_config} \
                        > "logs/${dataset_name}_${exp_name}_k${curvature}_lr${k_lr}_seed${seed}.txt" 2>&1
                    done
                done
            else
                # For Euclidean baseline
                echo "Running ${exp_name} with seed=${seed} on ${dataset_name}"
                CUDA_VISIBLE_DEVICES=${gpu_config} torchrun --standalone --nproc_per_node=2 \
                train_gpt2_main.py \
                --data_path "${path}" \
                --num_iterations "${NUM_ITERATIONS}" \
                --head_mode "${head_mode}" \
                --attn_mode "${attn_mode}" \
                --seed "${seed}" \
                ${model_config} \
                > "logs/${dataset_name}_${exp_name}_seed${seed}.txt" 2>&1
            fi
        done
    done
}

# Run all experiments
run_experiment "euc_baseline" "euc" "euc" "1,2" ""
run_experiment "hyp_head" "hyp" "euc" "1,2" "hyp"
run_experiment "hyp_attn" "euc" "hyp" "1,2" "hyp"
run_experiment "full_hyp" "hyp" "hyp" "1,2" "hyp"

echo "All experiments completed!"