#!/bin/bash

# Define a small set of parameters for testing
curvatures=(1000.0)  # Use a single curvature for the test
seeds=(1)        # Use a single seed for the test

# Test Hyperbolic training
echo "Testing hyperbolic model training with curvature=${curvatures[0]} and seed=${seeds[0]}..."
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 train_gpt2_hyp.py --lm_head 'hyp' --curvature "${curvatures[0]}" --seed "${seeds[0]}" > test_logs_hyperbolic.txt 2>&1

if [ $? -eq 0 ]; then
    echo "Hyperbolic model training test passed. Logs saved to test_logs_hyperbolic.txt"
else
    echo "Hyperbolic model training test failed. Check test_logs_hyperbolic.txt for details."
    exit 1
fi

echo "All tests passed successfully!"
