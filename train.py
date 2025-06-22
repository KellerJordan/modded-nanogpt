import os
import sys

# Configure torch inductor for flex_attention
import torch._inductor.config
torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON"

code = open(sys.argv[0]).read()
code += open('optimizer.py', 'r', encoding='utf-8').read()
code += open('dataloading.py', 'r', encoding='utf-8').read()
code += open('model/utils.py', 'r', encoding='utf-8').read()
code += open('model/attention.py', 'r', encoding='utf-8').read()
code += open('model/model.py', 'r', encoding='utf-8').read()

import uuid
import time
import contextlib
import math
import torch
import torch.distributed as dist
import argparse
from dataclasses import dataclass
from typing import Optional
from transformers import EsmTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

from optimizer import Muon
from dataloading import DistributedPaddedDataLoader
from model.model import PLM, PLMConfig
from model.utils import Linear


@dataclass
class TrainingArguments:
    # Model hyperparams
    hidden_size: int = 768
    num_attention_heads: int = 6
    num_hidden_layers: int = 12
    num_att_tokens: int = 512
    vocab_size: int = 33
    expansion_ratio: float = 8/3
    soft_logit_cap: float = 16.0
    p_attention: bool = False
    tie_embeddings: bool = False
    unet: bool = True

    # Data hyperparams
    input_bin: str = 'data/omgprot50/omgprot50_train_*.bin'
    input_valid_bin: str = 'data/omgprot50/omgprot50_valid_*.bin'
    input_test_bin: str = 'data/omgprot50/omgprot50_test_*.bin'

    # Optimization hyperparams
    batch_size: int = 8192
    grad_accum: int = 1
    num_steps: int = 20000
    cooldown_steps: int = 5000
    max_length: int = 2048

    # adam
    lr_embed: float = 0.06
    lr_head: float = 0.0008
    lr_scalar: float = 0.04
    # muon
    lr_hidden: float = 0.05
    muon_momentum_warmup_steps: int = 300  # steps for warmup momentum, 0.85 -> 0.95

    # Evaluation and logging hyperparams
    valid_loss_every: int = 500
    hf_model_name: Optional[str] = None
    save_every: Optional[int] = None


def get_param_count(model):
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
    return total_params


def main(args, model_config):
    # set up DDP (distributed data parallel) if available, otherwise single GPU
    # requires torchrun or equivalent
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

    print0(f'Model config: {model_config}')
    print0(f'Args: {args.__dict__}')

    # calculate the steps of gradient accumulation required to attain the desired global batch size
    # args.batch_size should refer to the total amount of tokens per backward pass
    # reducing batch_size by ddp_world_size is done in the data loader
    batch_size = args.batch_size // args.grad_accum

    print0(f'Train accumulation steps: {args.grad_accum}')
    print0(f'Adjusted local batch size: {batch_size} tokens')
    print0(f'Across {ddp_world_size} GPUs')
    print0(f'Total batch size: {args.batch_size} tokens')

    # load tokens
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    cls_id, eos_id, pad_id = tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    train_loader = DistributedPaddedDataLoader(args.input_bin, batch_size, ddp_rank, ddp_world_size,
                                               cls_id=cls_id, eos_id=eos_id, pad_id=pad_id, max_epochs=100)
    valid_loader = DistributedPaddedDataLoader(args.input_valid_bin, batch_size, ddp_rank, ddp_world_size,
                                               cls_id=cls_id, eos_id=eos_id, pad_id=pad_id, max_epochs=1)
    test_loader = DistributedPaddedDataLoader(args.input_test_bin, batch_size, ddp_rank, ddp_world_size,
                                              cls_id=cls_id, eos_id=eos_id, pad_id=pad_id, max_epochs=1)
    print0(f'Training DataLoader: {len(train_loader.files)} files')
    print0(f'Validation DataLoader: {len(valid_loader.files)} files')
    print0(f'Testing DataLoader: {len(test_loader.files)} files')
    print0('='*100, logonly=True)

    model = PLM(model_config)
    print(model)
    model = model.cuda().bfloat16()
    for m in model.modules():
        if isinstance(m, Linear):
            m.float()
    # model = torch.compile(model)  # Temporarily disabled due to flex_attention backward issues

    # wrap model in DDP only if using distributed training
    if ddp_world_size > 1:
        model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
        raw_model = model.module
    else:
        raw_model = model

    # init the optimizers
    hidden_matrix_params = [
        p for n, p in model.named_parameters() 
        if p.ndim >= 2 and "embed" not in n.lower() and "lm_head" not in n.lower() and p.requires_grad
    ]
    embed_params = [
        p for n, p in model.named_parameters() if "embed" in n.lower() and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters() if "lm_head" in n.lower() and p.requires_grad
    ]
    scalar_params = [
        p for n, p in model.named_parameters() 
        if p.ndim < 2 and "embed" not in n.lower() and "lm_head" not in n.lower() and p.requires_grad
    ]

    # init the optimizer(s)
    optimizer1 = torch.optim.Adam([
        dict(params=embed_params, lr=args.lr_embed),
        dict(params=head_params, lr=args.lr_head),
        dict(params=scalar_params, lr=args.lr_scalar)
    ], betas=(0.8, 0.95), fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=args.lr_hidden, momentum=0.95)
    optimizers = [optimizer1, optimizer2]


    # learning rate decay scheduler (linear warmup and cooldown)
    def get_lr(it):
        assert it <= args.num_steps
        # 1) constant lr for a while
        if it < args.num_steps - args.cooldown_steps:
            return 1.0
        # 2) linear cooldown
        else:
            decay_ratio = (args.num_steps - it) / args.cooldown_steps
            return decay_ratio


    class LerpTensor:
        def __init__(self, start_val, end_val, precision):
            self.start, self.end, self.prec = start_val, end_val, precision
            self.prev_val = None
            dtype = torch.int32 if isinstance(precision, int) else torch.float
            self.gpu_val = torch.tensor(0, dtype=dtype, device="cuda")

        def __call__(self, frac_done):
            val = ((1 - frac_done) * self.start + frac_done * self.end) // self.prec * self.prec
            if val != self.prev_val:
                self.gpu_val.copy_(val, non_blocking=True)
                self.prev_val = val
            return self.gpu_val

    lerp_sw_size = LerpTensor(start_val=512, end_val=args.max_length, precision=128)

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

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
        # TODO
        # We should add this before the hackathon
        if step == 10:
            training_time_ms = 0
            t0 = time.perf_counter()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        frac_done = step / args.num_steps  # training progress
        sliding_window_size = lerp_sw_size(frac_done)

        # once in a while evaluate the validation dataset
        if args.valid_loss_every > 0 and step % args.valid_loss_every == 0 or last_step:
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            # run validation batches
            model.eval()
            valid_loader.reset()
            val_loss, valid_tokens = 0.0, 0

            with torch.no_grad():
                input_ids = valid_loader.next_batch()
                while input_ids.numel():
                    batch_valid_tokens = (input_ids != pad_id).sum()
                    valid_tokens += batch_valid_tokens
                    val_loss += model(input_ids, sliding_window_size).loss * batch_valid_tokens
                    input_ids = valid_loader.next_batch()
            
            if ddp_world_size > 1:
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
            
            val_loss /= valid_tokens
            # log val loss to console and to logfile
            print0(f'step:{step}/{args.num_steps} \
                   val_loss:{val_loss:.4f} \
                   train_time:{training_time_ms:.0f}ms \
                   step_avg:{training_time_ms/(timed_steps-1):.2f}ms \
                   perplexity:{(math.e**val_loss):.4f} \
                   param_count:{get_param_count(model):,} \
                   tokens: {valid_tokens.item():,}')
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
                torch.save(log, 'logs/state_step%06d.pt' % step)

                if args.hf_model_name:
                    try:
                        if ddp_world_size > 1:
                            model.module.push_to_hub(args.hf_model_name, subfolder='step%06d' % step)
                        else:
                            model.push_to_hub(args.hf_model_name, subfolder='step%06d' % step)
                    except Exception as e:
                        print0(e)

                torch.cuda.synchronize()
                t0 = time.perf_counter()

        if last_step:
            break

        # --------------- FORWARD AND BACKWARD PASS -----------------
        model.train()
        for i in range(args.grad_accum):
            with contextlib.ExitStack() as stack:
                # Only sync gradients on last accumulation step
                if ddp_world_size > 1 and i < args.grad_accum - 1:
                    stack.enter_context(model.no_sync())
                input_ids = train_loader.next_batch()
                (model(input_ids, sliding_window_size).loss / args.grad_accum).backward()

        # momentum warmup for Muon
        frac = min(step/args.muon_momentum_warmup_steps, 1)
        for group in optimizer2.param_groups:
            group['momentum'] = (1 - frac) * 0.85 + frac * 0.95

        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()

        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- FORWARD AND BACKWARD PASS END -------------------
        # everything that follows now is just eval, diagnostics, prints, logging, etc.
        if step % 100 == 0:
            approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(f'step:{step+1}/{args.num_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms')

    print0(f'peak memory consumption training: {torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024} GiB')
    print0(f'Train Time: {training_time_ms:.0f}ms | Step Avg: {training_time_ms/(timed_steps-1):.2f}ms | Param Count: {get_param_count(model):,}')
    print0(f'Total train time (min): {training_time_ms / 60000:.2f}')
    print0(f'Total train time (hours): {training_time_ms / 3600000:.2f}')
    # save the model to huggingface
    try:
        if ddp_world_size > 1:
            model.module.push_to_hub(args.hf_model_name)
        else:
            model.push_to_hub(args.hf_model_name)
    except Exception as e:
        print(e)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.manual_seed(42)
    model.eval()
    test_loader.reset()

    test_loss, test_tokens = 0.0, 0
    with torch.no_grad():
        input_ids = test_loader.next_batch()
        while input_ids.numel():
            batch_test_tokens = (input_ids != pad_id).sum()
            test_tokens += batch_test_tokens
            test_loss += model(input_ids, sliding_window_size).loss * batch_test_tokens
            input_ids = test_loader.next_batch()
    
    if ddp_world_size > 1:
        dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_tokens, op=dist.ReduceOp.SUM)
    
    test_loss /= test_tokens
    print0(f'Test tokens: {test_tokens.item()}')
    print0(f'Loss: {test_loss:.4f} | Perplexity: {math.e**test_loss:.4f}')
    print0(f"peak memory consumption testing: {torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024} GiB")
    # -------------------------------------------------------------------------
    # clean up nice
    if ddp_world_size > 1:
        dist.destroy_process_group()
    return val_loss, test_loss


def arg_parser():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--save_path", type=str, default="Synthyra/speedrun_test", help="Path to save the model and report to wandb")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = arg_parser()
    args = TrainingArguments()

    if cli_args.bugfix:
        args.hidden_size = 128
        args.num_attention_heads = 2
        args.num_hidden_layers = 2
        args.num_att_tokens = 128
        args.expansion_ratio = 2.0
        args.soft_logit_cap = 16.0
        args.p_attention = False
        args.tie_embeddings = False
        args.unet = True
        args.batch_size = 2048
        args.grad_accum = 1
        args.num_steps = 10
        args.cooldown_steps = 2
        args.max_length = 512
        args.lr_embed = 0.06
        args.lr_head = 0.0008

    model_config = PLMConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        num_att_tokens=args.num_att_tokens,
        vocab_size=args.vocab_size,
        expansion_ratio=args.expansion_ratio,
        soft_logit_cap=args.soft_logit_cap,
        p_attention=args.p_attention,
        tie_embeddings=args.tie_embeddings,
        unet=args.unet,
    )

    if cli_args.token:
        from huggingface_hub import login
        login(cli_args.token)
        cli_args.token = None
    
    val_loss, test_loss = main(args, model_config)
