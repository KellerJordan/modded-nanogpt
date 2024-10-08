torchrun --standalone --nproc_per_node=8 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --model d12 \
    --batch_size 64 \
    --sequence_length 1024 \
    --val_loss_every 128 \
    --num_iterations 6676 \
    --weight_decay 0.0 \
    --learning_rate 0.0036 \
    --warmup_iters 0 \
    --warmdown_iters 2000
