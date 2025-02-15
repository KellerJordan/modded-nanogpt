import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import random
import datetime
import time
from pathlib import Path
import torch
import torch.distributed as dist
import numpy as np

from utils.logging import setup_logging
from utils.distributed import setup_distributed
from utils.optimization import setup_optimizers
from utils.config import parse_arguments, FullConfig
from data.loader import setup_data
from model.setup import setup_model

def validate(model, val_loader, val_steps, ctx):
    """Run validation loop."""
    model.eval()
    val_loader.reset()
    val_loss = 0.0
    
    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()
        with ctx:
            _, loss = model(x_val, y_val, return_logits=False)
            val_loss += loss.detach()
            del loss
            
    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    return val_loss / val_steps

def train_step(model, x, y, optimizers, schedulers, train_accumulation_steps, ctx):
    """Execute single training step with gradient accumulation."""
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        
        # Gradient accumulation
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    # Scale gradients
    for name, p in model.named_parameters():
        if p.grad is not None:
            p.grad /= train_accumulation_steps

    # Optimization step
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    
    model.zero_grad(set_to_none=True)
    return train_loss

def main():
    """Main training function."""
    # Parse arguments and create config
    args = parse_arguments()
    config = FullConfig(**vars(args))
    
    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    # Setup distributed training
    ddp_rank, ddp_world_size, device = setup_distributed()
    master_process = (ddp_rank == 0)
    
    # Setup model, data, and logging
    model, raw_model = setup_model(config, device, ddp_rank)
    train_loader, val_loader = setup_data(config, ddp_rank, ddp_world_size)
    writer, logfile, run_id = setup_logging(config, args, master_process)
    
    # Setup optimizers and schedulers
    optimizers, schedulers = setup_optimizers(model, config, master_process)
    
    # Training loop configuration
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    B, T = config.device_batch_size, config.sequence_length
    val_steps = config.val_tokens // (B * T * ddp_world_size)
    train_accumulation_steps = config.batch_size // (B * ddp_world_size)
    
    # Initialize training state
    training_time_s = 0.0
    torch.cuda.synchronize()
    t0 = time.time()
    total_t0 = time.time()
    train_loss_accum = 0.0
    train_loss_count = 0
    
    WARMUP_STEPS = 10
    train_loader.reset()
    x, y = train_loader.next_batch()
    
    # Main training loop
    for step in range(config.num_iterations + 1):
        last_step = (step == config.num_iterations)
        
        # Handle warmup period timing
        if step == WARMUP_STEPS:
            training_time_s = 0.0
            t0 = time.time()
        timed_steps = float('nan') if step <= (WARMUP_STEPS + 1) else (step - WARMUP_STEPS) + 1

        # Validation loop
        if last_step or (config.val_loss_every > 0 and step % config.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_s += time.time() - t0
            
            val_loss = validate(model, val_loader, val_steps, ctx)
            
            if master_process:
                tokens_seen = step * config.batch_size * config.sequence_length
                writer.add_scalar('Loss/Validation', val_loss.item(), tokens_seen)
            
            torch.cuda.synchronize()
            t0 = time.time()

        if last_step:
            break

        # Training step
        train_loss = train_step(
            model, x, y, optimizers, schedulers,
            train_accumulation_steps, ctx
        )
        x, y = train_loader.next_batch()
        
        # Accumulate training statistics
        train_loss_accum += train_loss.item()
        train_loss_count += 1

        # Log training metrics
        if master_process and (step+1) % config.train_loss_every == 0:
            avg_train_loss = train_loss_accum / train_loss_count
            tokens_seen = step * config.batch_size * config.sequence_length
            writer.add_scalar('Loss/Train', avg_train_loss, tokens_seen)
            train_loss_accum = 0.0
            train_loss_count = 0

    # Cleanup
    if master_process:
        writer.close()
    dist.destroy_process_group()

    if ddp_rank == 0:
        print("\n=== Relevant Hyperparameters ===")
        print(f"Data Path:            {config.data_path}")
        print(f"Sequence Length:      {config.sequence_length}")
        print(f"Batch Size (global):  {config.batch_size}")
        print(f"Batch Size (device):  {config.device_batch_size}")
        print(f"n_layers:             {config.n_layers}")
        print(f"n_heads:             {config.n_heads}")
        print(f"head_dim:            {config.head_dim}")
        print(f"n_embd:              {config.n_embd}")

if __name__ == "__main__":
    main()
