import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging

with open('optimizer.py', 'r', encoding='utf-8') as f:
    source_code = f.read()
    code += source_code

with open('model.py', 'r', encoding='utf-8') as f:
    source_code = f.read()
    code += source_code

with open('utils.py', 'r', encoding='utf-8') as f:
    source_code = f.read()
    code += source_code

with open('dataloading.py', 'r', encoding='utf-8') as f:
    source_code = f.read()
    code += source_code

import argparse
import uuid
import time
import contextlib
import math
import torch
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import EsmTokenizer
from pathlib import Path

from optimizer import Muon
from model import ModelConfig, ESM, CastedLinear
from dataloading import DistributedDataLoader, TestDataset


def get_args():
    parser = argparse.ArgumentParser(description='ESM2 training arguments')
    
    # Model hyperparams
    parser.add_argument('--vocab_size', type=int, default=33, help='vocabulary size')
    parser.add_argument('--num_layers', type=int, default=20, help='number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=6, help='number of attention heads (head dim 128 suggested by @Grad62304977)')
    parser.add_argument('--model_dim', type=int, default=768, help='model hidden dimension size')
    
    # Data hyperparams
    parser.add_argument('--input_bin', type=str, default='data/omgprot50/omgprot50_train_*.bin', help='input .bins to train on')
    parser.add_argument('--input_val_bin', type=str, default='data/omgprot50/omgprot50_valid_*.bin', help='input .bins to eval validation loss on')
    #parser.add_argument('--input_test_bin', type=str, default='data/omgprot50/omgprot50_test_*.bin', help='input .bins to eval test loss on')   
    
    # Optimization hyperparams
    parser.add_argument('--batch_size', type=int, default=64*1024, help='batch size, in tokens, across all devices')
    parser.add_argument('--grad_accum', type=int, default=1, help='manually set number of gradient accumulation steps, else, will be ddp_world_size')
    parser.add_argument('--num_steps', type=int, default=25000, help='number of iterations to run')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='number of warmup steps')
    parser.add_argument('--cooldown_steps', type=int, default=1000, help='number of cooldown steps')
    
    # Evaluation and logging hyperparams
    parser.add_argument('--val_loss_every', type=int, default=500, help='every how many steps to evaluate val loss? 0 for only at the end')
    parser.add_argument('--val_tokens', type=int, default=2077660, help='how many tokens of validation data? important to keep fixed for consistent comparisons')
    parser.add_argument('--save_every', type=int, default=None, help='save every how many steps? None for no saving')
    args = parser.parse_args()
    return args


def get_param_count(model):
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
    return total_params


if __name__ == "__main__":
    args = get_args()
    model_config = ModelConfig(args)

    # set up DDP (distributed data parallel) if available, otherwise single GPU
    if 'RANK' in os.environ:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{ddp_local_rank}')
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()
        master_process = (ddp_rank == 0)
    else:
        ddp_rank = 0
        ddp_local_rank = 0 
        ddp_world_size = 1
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        master_process = True

    print(f'using device: {device}')

    # begin logging
    logfile = None
    if master_process:
        run_id = uuid.uuid4()
        Path('logs').mkdir(exist_ok=True)
        # logdir = Path('logs') / f'{run_id}'
        # logdir.mkdir()
        logfile = Path('logs') / f'{run_id}.txt'
        print(logfile.stem)
        # create the log file
        with logfile.open('w') as f:
            # begin the log by printing this file (the Python code)
            print(code, file=f)
            print('=' * 100, file=f)

    def print0(s, logonly=False):
        if master_process:
            with logfile.open('a') as f:
                if not logonly:
                    print(s)
                print(s, file=f)

    # log information about the hardware/software environment this is running on
    # and print the full `nvidia-smi` to file
    print0(f'Running python {sys.version}')
    print0(f'Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:')
    import subprocess
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print0(f'{result.stdout}', logonly=True)
    print0('='*100, logonly=True)

    # calculate the number of steps to take in the val loop.
    args.val_tokens = (args.val_tokens // (args.batch_size * ddp_world_size)) * (args.batch_size * ddp_world_size)
    val_steps = args.val_tokens // (args.batch_size * ddp_world_size)
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    train_accumulation_steps = 1
    batch_size = args.batch_size
    if ddp_world_size > 1:
        assert args.batch_size % ddp_world_size == 0
        train_accumulation_steps = ddp_world_size
        batch_size = args.batch_size // ddp_world_size
    
    if args.grad_accum > 1:
        train_accumulation_steps *= args.grad_accum
        batch_size = args.batch_size // args.grad_accum

    print0(f'Train accumulation steps: {train_accumulation_steps}')
    print0(f'Adjusted local batch size: {batch_size} tokens')
    print0(f'Across {ddp_world_size} GPUs')
    print0(f'Total batch size: {args.batch_size} tokens')

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, batch_size, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, batch_size, ddp_rank, ddp_world_size)
    print0(f"Training DataLoader: total number of tokens: {train_loader.total_num_tokens} across {len(train_loader.files)} files")
    print0(f"Validation DataLoader: total number of tokens: {val_loader.total_num_tokens} across {len(val_loader.files)} files")
    print0('='*100, logonly=True)
    input_ids = train_loader.next_batch()

    model = ESM(model_config)
    model = model.cuda().bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    config.coordinate_descent_tuning = True # suggested by @Chillee
    model = torch.compile(model)

    # wrap model in DDP only if using distributed training
    if ddp_world_size > 1:
        model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
        raw_model = model.module
    else:
        raw_model = model

    # init the optimizers
    embed_params = [*raw_model.embed.parameters(), *raw_model.value_embeds.parameters()]
    optimizer1 = torch.optim.Adam(embed_params, lr=0.6, betas=(0.8, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.008, betas=(0.8, 0.95), fused=True)
    params = list(raw_model.blocks.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
    optimizer3 = Muon(matrix_params, lr=0.05, momentum=0.95)
    optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True)
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

    # learning rate decay scheduler (linear warmup and cooldown)
    def get_lr(it):
        assert it <= args.num_steps
        # 1) linear warmup for warmup_steps steps
        if it < args.warmup_steps:
            return (it+1) / args.warmup_steps
        # 2) constant lr for a while
        elif it < args.num_steps - args.cooldown_steps:
            return 1.0
        # 3) linear cooldown
        else:
            decay_ratio = (args.num_steps - it) / args.cooldown_steps
            return decay_ratio

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    sliding_window_size = torch.tensor(1024 - 128, dtype=torch.int32, device="cuda")
    sw_prev = 1024 - 128
    # Start training loop
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()


    ### BEGIN TRAINING LOOP ###
    for step in range(args.num_steps + 1):
        last_step = (step == args.num_steps)
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.perf_counter()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        # Linearly increase the sliding window size over training in chunks of 128 from 1024 -> 2048. By @fernbear.bsky.social
        frac_done = step / args.num_steps # training progress
        sw_size = int(((1 - frac_done) * 1023 + frac_done * 2048) // 128) * 128
        if sw_size != sw_prev:
            sliding_window_size.copy_(sw_size, non_blocking=True)
            sw_prev = sw_size

        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and step % args.val_loss_every == 0:
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            # run validation batches
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(val_steps):
                    seq_val = val_loader.next_batch()
                    val_loss += model(seq_val, sliding_window_size)
            if ddp_world_size > 1:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            # log val loss to console and to logfile
            print0(f'step:{step}/{args.num_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms perplexity:{(math.e**val_loss):.4f} param_count:{get_param_count(model):,}')
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # save checkpoint every `save_every` steps
        if master_process and args.save_every:
            if last_step or (step % args.save_every == 0):
                # stop the clock
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.perf_counter() - t0)
                # save the state of the training process
                log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
                # start the clock again
                torch.cuda.synchronize()
                t0 = time.perf_counter()

        if last_step:
            break

        # --------------- FORWARD AND BACKWARD PASS -----------------
        model.train()
        for i in range(1, train_accumulation_steps + 1):
            with contextlib.ExitStack() as stack:
                if ddp_world_size > 1 and i < train_accumulation_steps: # there's no need to sync gradients every accumulation step
                    stack.enter_context(model.no_sync())
                #if step >= 5:
                #    stack.enter_context(torch.compiler.set_stance(skip_guard_eval_unsafe=True))
                model(input_ids, sliding_window_size).backward()
                input_ids = train_loader.next_batch()
        if train_accumulation_steps != 1:
            for p in model.parameters():
                p.grad /= train_accumulation_steps
        # momentum warmup for Muon
        frac = min(step/300, 1)
        for group in optimizer3.param_groups:
            group['momentum'] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- FORWARD AND BACKWARD PASS END -------------------
        # everything that follows now is just eval, diagnostics, prints, logging, etc.
        approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{args.num_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024} GiB")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.manual_seed(42)
    import numpy as np
    from torch.utils.data import DataLoader
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
    model.eval()
    results, all_true, all_pred = [], [], []
    total_loss, count = 0.0, 0
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    test_loader = DataLoader(TestDataset(tokenizer, batch_size // 4), batch_size=1) # was getting ooms so reducing batch size

    from tqdm import tqdm
    with torch.no_grad():
        for input_ids, labels in tqdm(test_loader, total=len(test_loader), desc="Evaluating"):
            loss, logits = model.inference(input_ids.cuda(), labels.cuda())        
            total_loss += loss.detach().cpu().item()
            count += 1
            all_true.extend(labels.cpu().numpy().flatten())
            all_pred.extend(logits.argmax(dim=-1).cpu().numpy().flatten())

    average_loss = total_loss / count
    perplexity = torch.exp(torch.tensor(average_loss)).item()
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    mask = (all_true != -100)
    all_true = all_true[mask]
    all_pred = all_pred[mask]

    precision = precision_score(all_true, all_pred, average='weighted')
    recall = recall_score(all_true, all_pred, average='weighted')
    f1 = f1_score(all_true, all_pred, average='weighted')
    accuracy = accuracy_score(all_true, all_pred)
    mcc = matthews_corrcoef(all_true, all_pred)

    print0("Final Results:")
    print0(f"  Loss:        {average_loss:.4f}")
    print0(f"  Perplexity:  {perplexity:.4f}")
    print0(f"  Precision:   {precision:.4f}")
    print0(f"  Recall:      {recall:.4f}")
    print0(f"  F1:          {f1:.4f}")
    print0(f"  Accuracy:    {accuracy:.4f}")
    print0(f"  MCC:         {mcc:.4f}")

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp_world_size > 1:
        dist.destroy_process_group()
