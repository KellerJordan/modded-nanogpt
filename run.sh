# Yields 3.2798 perplexity in 6.44B tokens.
# For comparison, the llm.c trainer gets 3.2847 in 10B tokens, which is GPT-2 level quality.
# The gain in efficiency over this baseline is due to the following changes:
# 1. Increased learning rate
# 2. Halved batch size (~but same training speed)
# 3. Improved learning rate schedule (linear up from 0, then linear down to 0.1 * max)
# 4. Normalized the gradient for each weight to have unit norm
# 5. Removed all affine scale and bias parameters from the architecture, and switched to
#    RMSNorm (actually this just simplifies the code but doesn't speed up training)

torchrun --standalone --nproc_per_node=8 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir pylog124M \
    --model d12 \
    --batch_size 32 \
    --sequence_length 1024 \
    --total_batch_size 262144 \
    --val_loss_every 128 \
    --num_iterations 24576 \
    --weight_decay 0.1 \
    --learning_rate 0.0015 \
    --warmup_iters 256

