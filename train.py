import os
import sys
import yaml

code = open(sys.argv[0]).read()
code += open('optimizer.py', 'r', encoding='utf-8').read()
code += open('data/dataloading.py', 'r', encoding='utf-8').read()
code += open('model/utils.py', 'r', encoding='utf-8').read()
code += open('model/attention.py', 'r', encoding='utf-8').read()
code += open('model/model.py', 'r', encoding='utf-8').read()

import uuid
import time
import contextlib
import subprocess
import math
import random
import numpy as np
import argparse
import torch
import torch.distributed as dist
#import torch._dynamo
import torch._inductor.config as inductor_config
from torchinfo import summary
from transformers import EsmTokenizer, get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from tqdm import tqdm

from model.model import PLM, PLMConfig
from model.utils import Linear
from data.dataloading import OptimizedTrainLoader, OptimizedEvalLoader
from optimizer import Muon
from utils import LerpTensor


global WANDB_AVAILABLE
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


#torch._dynamo.config.suppress_errors = True
inductor_config.max_autotune_gemm_backends = "ATEN,CUTLASS,FBGEMM"


def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}


def arg_parser():
    parser = argparse.ArgumentParser(description="Synthyra Trainer")
    parser.add_argument("--yaml_path", type=str, default=None, help="Path to YAML file")

    # CLI-specific arguments (always from CLI for security)
    parser.add_argument("--token", type=str, default=None, help="Huggingface token")
    parser.add_argument("--wandb_token", type=str, default=None, help="Weights & Biases API token")
    parser.add_argument("--log_name", type=str, default=None, help="Name of the log file, else will be randomly generated")
    parser.add_argument("--bugfix", action="store_true", help="Use small batch size and max length for debugging")
    
    # All other arguments with defaults (can be overridden by YAML)
    parser.add_argument("--save_path", type=str, default="Synthyra/speedrun_test", help="Path to save the model and report to wandb")
    
    # Distributed training arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--clear_cache_every", type=int, default=1000, help="Clear CUDA cache every N steps")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping value (0 to disable)")
    
    # Model hyperparams
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size of the model")
    parser.add_argument("--num_attention_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--num_hidden_layers", type=int, default=24, help="Number of hidden layers")
    parser.add_argument("--num_att_tokens", type=int, default=512, help="Number of attention tokens")
    parser.add_argument("--vocab_size", type=int, default=33, help="Vocabulary size")
    parser.add_argument("--expansion_ratio", type=float, default=8/3, help="Expansion ratio for MLP")
    parser.add_argument("--soft_logit_cap", type=float, default=32.0, help="Soft logit cap")
    parser.add_argument("--attention_soft_cap", type=float, default=64.0, help="Attention softmax cap")
    parser.add_argument("--add_att_soft_cap", type=bool, default=True, help="Add attention softmax cap")
    parser.add_argument("--p_attention", action="store_true", help="Use P attention")
    parser.add_argument("--tie_embeddings", action="store_true", help="Tie embeddings")
    parser.add_argument("--unet", type=bool, default=True, help="Use UNet architecture")
    
    # Data hyperparams
    parser.add_argument("--input_bin", type=str, default='data/omgprot50/omgprot50_train_*.bin', help="Input training bin files pattern")
    parser.add_argument("--input_valid_bin", type=str, default='data/omgprot50/omgprot50_valid_*.bin', help="Input validation bin files pattern")
    parser.add_argument("--input_test_bin", type=str, default='data/omgprot50/omgprot50_test_*.bin', help="Input test bin files pattern")
    parser.add_argument("--mlm", type=bool, default=False, help="Use masked language modeling")
    parser.add_argument("--mask_rate", type=float, default=0.2, help="Mask rate for masked language modeling")
    
    # Optimization hyperparams
    parser.add_argument("--batch_size", type=int, default=8*64*1024, help="Total batch size in tokens")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--cooldown_steps", type=int, default=5000, help="Number of cooldown steps")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--scheduler_type", type=str, default='cosine', help="Scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000, help="Number of warmup steps")

    # Adam optimizer params
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer when not using Muon")
    parser.add_argument("--lr_embed", type=float, default=0.06, help="Learning rate for embeddings")
    parser.add_argument("--lr_head", type=float, default=0.008, help="Learning rate for head")
    parser.add_argument("--lr_scalar", type=float, default=0.04, help="Learning rate for scalar params")
    
    # Muon optimizer params
    parser.add_argument("--use_muon", type=bool, default=True, help="Use Muon optimizer")
    parser.add_argument("--lr_hidden", type=float, default=0.05, help="Learning rate for hidden layers (Muon)")
    parser.add_argument("--muon_momentum_warmup_steps", type=int, default=300, help="Steps for warmup momentum (0.85 -> 0.95)")
    
    # Evaluation and logging hyperparams
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate on validation set every N steps")
    parser.add_argument("--hf_model_name", type=str, default='lhallee/speedrun', help="Huggingface model name for saving")
    parser.add_argument("--save_every", type=int, default=None, help="Save checkpoint every N steps")
    
    # Dataloader params
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for optimized dataloader")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor for optimized dataloader")
    
    # Parse CLI args first
    args = parser.parse_args()
    
    # Load YAML config if provided
    if args.yaml_path:
        yaml_config = load_config_from_yaml(args.yaml_path)
        
        # Security: Never load tokens from YAML files
        cli_only_params = {'token', 'wandb_token', 'yaml_path'}
        
        # Override defaults with YAML values, but preserve CLI overrides
        for key, value in yaml_config.items():
            if key not in cli_only_params and hasattr(args, key):
                # Only override if the argument wasn't explicitly provided via CLI
                # Check if the current value is the default by comparing with parser defaults
                action = next((action for action in parser._actions if action.dest == key), None)
                if action and getattr(args, key) == action.default:
                    # Convert boolean strings to boolean values
                    if isinstance(action.default, bool) and isinstance(value, str):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    setattr(args, key, value)
    
    return args


def set_seed(seed):
    """Set seed for reproducibility across all processes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class GlobalTimer:
    """Global timer that tracks elapsed time and can be paused/resumed."""
    def __init__(self):
        self.total_time = 0.0
        self.start_time = None
        self.is_running = False
    
    def start(self):
        """Start the timer."""
        if not self.is_running:
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()
            self.is_running = True
    
    def pause(self):
        """Pause the timer and add elapsed time to total."""
        if self.is_running:
            torch.cuda.synchronize()
            self.total_time += time.perf_counter() - self.start_time
            self.is_running = False
    
    def resume(self):
        """Resume the timer."""
        self.start()
    
    def get_time(self):
        """Get total elapsed time including current session if running."""
        current_time = self.total_time
        if self.is_running:
            torch.cuda.synchronize()
            current_time += time.perf_counter() - self.start_time
        return current_time
    
    def reset(self):
        """Reset the timer to zero."""
        self.total_time = 0.0
        self.start_time = None
        self.is_running = False


def exclude_from_timer(timer):
    """Decorator that pauses the timer during function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer.pause()
            try:
                result = func(*args, **kwargs)
            finally:
                timer.resume()
            return result
        return wrapper
    return decorator


class Trainer:
    def __init__(self, args, model_config):
        self.args = args
        self.model_config = model_config
        
        # Initialize global timer
        self.train_timer = GlobalTimer()
        
        if 'RANK' in os.environ:
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = torch.device(f'cuda:{self.ddp_local_rank}')
            torch.cuda.set_device(self.device)
            dist.init_process_group(backend='nccl', device_id=self.device)
            dist.barrier()
            self.master_process = (self.ddp_rank == 0)
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)
            self.master_process = True

        set_seed(self.args.seed)
        
        print(f'Process {self.ddp_rank}: using device: {self.device}')

    def print0(self, s, logonly=False):
        if self.master_process:
            with self.logfile.open('a', encoding='utf-8') as f:
                if not logonly:
                    print(s)
                print(s, file=f)
    
    def log_wandb(self, log_dict, prefix='train'):
        if self.master_process and self.args.wandb_token:
            wandb.log({f'{prefix}/{k}': v for k, v in log_dict.items()})

    def init_training(self):
        self.logfile = None
        if self.master_process:
            Path('logs').mkdir(exist_ok=True)
            
            # Use provided log_name or generate a random UUID
            if self.args.log_name:
                log_filename = f'{self.args.log_name}.txt'
            else:
                run_id = uuid.uuid4()
                log_filename = f'{run_id}.txt'
                
            self.logfile = Path('logs') / log_filename
            print(self.logfile.stem)
            # create the log file
            with self.logfile.open('w', encoding='utf-8') as f:
                # begin the log by printing this file (the Python code)
                print(code, file=f)
                print('=' * 100, file=f)

        # Synchronize before initializing wandb
        if self.ddp_world_size > 1:
            dist.barrier()

        if self.master_process and self.args.wandb_token:
            wandb.login(key=self.args.wandb_token)
            wandb.init(
                project="speedrunning-esm2",
                name=self.args.save_path,
                config={
                    **vars(self.args),
                    **vars(self.model_config),
                    "ddp_world_size": self.ddp_world_size,
                    "device": str(self.device)
                }
            )

        self.print0(f'Running python {sys.version}')
        self.print0(f'Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:')
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.print0(f'{result.stdout}', logonly=True)
        self.print0('='*100, logonly=True)

        # Log configuration source
        if self.args.yaml_path:
            self.print0(f'Configuration loaded from YAML: {self.args.yaml_path}')
            self.print0('CLI arguments override YAML where provided (tokens always from CLI for security)')
        else:
            self.print0('Configuration from CLI arguments only')
        self.print0('='*50)
        
        self.print0(f'Model config:\n{self.model_config}')
        self.print0('Args:')
        for k, v in self.args.__dict__.items():
            self.print0(f'{k}: {v}')
        self.print0('='*100, logonly=True)

        # calculate local batch size
        self.batch_size = self.args.batch_size // self.args.grad_accum // self.ddp_world_size

        self.print0(f'Train accumulation steps: {self.args.grad_accum}')
        self.print0(f'Adjusted local batch size: {self.batch_size} tokens')
        self.print0(f'Across {self.ddp_world_size} GPUs')
        self.print0(f'Total batch size: {self.args.batch_size} tokens')

        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.pad_token_id = self.tokenizer.pad_token_id

        self.train_loader = self.init_dataloader(self.args.input_bin, training=True)
        self.valid_loader = self.init_dataloader(self.args.input_valid_bin, training=False)
        self.test_loader = self.init_dataloader(self.args.input_test_bin, training=False)

        self.print0(f'Training DataLoader: {len(self.train_loader.files)} files')
        self.print0(f'Validation DataLoader: {len(self.valid_loader.files)} files')
        self.print0(f'Testing DataLoader: {len(self.test_loader.files)} files')
        self.print0('='*100, logonly=True)

        self.model = self.init_model()
        self.print0(summary(self.model))
        self.optimizers = self.init_optimizers()
        self.lr_schedulers, self.sliding_window_size_scheduler = self.init_schedulers()
        self.print0(f"Ready for training!")
        
        # Create decorated versions of methods that should be excluded from timing
        self._run_eval_loader_timed = exclude_from_timer(self.train_timer)(self.run_eval_loader)
        self._save_checkpoint_timed = exclude_from_timer(self.train_timer)(self.save_checkpoint)

    def init_dataloader(self, filename_pattern, training=True):
        if training:
            return OptimizedTrainLoader(
                filename_pattern=filename_pattern,
                seq_len=self.batch_size,
                process_rank=self.ddp_rank,
                num_processes=self.ddp_world_size,
                max_epochs=1,
                tokenizer=self.tokenizer,
                num_workers=self.args.num_workers,
                prefetch_factor=self.args.prefetch_factor,
                mlm=self.args.mlm,
                mask_rate=self.args.mask_rate,
            )
        else:
            # Use evaluation dataloader that distributes data by sequences, not files
            return OptimizedEvalLoader(
                filename_pattern=filename_pattern,
                seq_len=self.batch_size,
                process_rank=self.ddp_rank,
                num_processes=self.ddp_world_size,
                tokenizer=self.tokenizer,
            )

    def init_model(self):
        self.print0("Initializing model...")
        model = PLM(self.model_config)
        self.print0(model)
        model = model.cuda().bfloat16()
        for m in model.modules():
            if isinstance(m, Linear):
                m.float()
        
        # Synchronize before compilation
        if self.ddp_world_size > 1:
            dist.barrier()
        
        self.print0("Coordinate descent tuning - can take up to 30 minutes")
        inductor_config.coordinate_descent_tuning = True
        self.print0("Calling torch.compile()")
        # Enable scalar output capture for .item() calls in compiled functions
        #torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model)
        
        if self.ddp_world_size > 1:
            # Use static graph if model architecture doesn't change
            model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
        return model

    def init_optimizers(self):
        self.print0("Initializing optimizers...")
        if self.args.use_muon:
            hidden_matrix_params = [
                p for n, p in self.model.named_parameters() 
                if p.ndim >= 2 and "embed" not in n.lower() and "lm_head" not in n.lower() and p.requires_grad
            ]
            embed_params = [
                p for n, p in self.model.named_parameters() if "embed" in n.lower() and p.requires_grad
            ]
            head_params = [
                p for n, p in self.model.named_parameters() if "lm_head" in n.lower() and p.requires_grad
            ]
            scalar_params = [
                p for n, p in self.model.named_parameters() 
                if p.ndim < 2 and "embed" not in n.lower() and "lm_head" not in n.lower() and p.requires_grad
            ]
            optimizer1 = torch.optim.Adam([
                dict(params=embed_params, lr=self.args.lr_embed),
                dict(params=head_params, lr=self.args.lr_head),
                dict(params=scalar_params, lr=self.args.lr_scalar)
            ], betas=(0.8, 0.95), fused=True)
            optimizer2 = Muon(hidden_matrix_params, lr=self.args.lr_hidden, momentum=0.95)
            optimizers = [optimizer1, optimizer2]
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
            optimizers = [optimizer]
        return optimizers
    
    def init_schedulers(self):
        self.print0("Initializing schedulers...")
        lr_schedulers = []
        adam_scheduler = get_scheduler(
            self.args.scheduler_type,
            optimizer=self.optimizers[0],
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.num_steps
        )
        lr_schedulers.append(adam_scheduler)
        if self.args.use_muon:
            muon_scheduler = get_scheduler(
                self.args.scheduler_type,
                optimizer=self.optimizers[-1],
                num_warmup_steps=0, # apparently muon does not need a warmup
                num_training_steps=self.args.num_steps
            )
            lr_schedulers.append(muon_scheduler)
        sliding_window_size_scheduler = LerpTensor(start_val=512, end_val=self.args.max_length, precision=128)
        return lr_schedulers, sliding_window_size_scheduler

    @torch.no_grad()
    def run_eval_loader(self, loader, prefix='val'): # returns loss, tokens
        # Synchronize before evaluation
        if self.ddp_world_size > 1:
            dist.barrier()
            
        loader.reset()
        self.model.eval()

        losses, total_tokens = [], 0
        input_ids, labels, mask_rate = loader.next_batch()
        
        # Only show progress bar on master process
        pbar = tqdm(desc=f'{prefix} set', leave=False, disable=not self.master_process)
        
        while input_ids.numel():
            batch_valid_tokens = (input_ids != self.pad_token_id).sum()
            total_tokens += batch_valid_tokens
            loss = self.model(input_ids, labels, mask_rate, self.sliding_window_size)
            losses.append(loss.item())
            input_ids, labels, mask_rate = loader.next_batch()
            pbar.update(1)
        pbar.close()

        avg_loss = sum(losses) / len(losses) if losses else 0.0

        if self.ddp_world_size > 1:
            # Convert to tensors before all_reduce
            avg_loss = torch.tensor(avg_loss, device=self.device)
            total_tokens = torch.tensor(total_tokens, device=self.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
            # Ensure all processes finish evaluation
            dist.barrier()

        perplexity = math.e**avg_loss if isinstance(avg_loss, float) else math.e**avg_loss.item()

        self.print0(f'{prefix} set: loss: {avg_loss:.4f} perplexity: {perplexity:.4f} tokens: {total_tokens.item() if hasattr(total_tokens, "item") else total_tokens:,}')

        return avg_loss, perplexity, total_tokens

    def save_checkpoint(self, step):
        # Only master saves, but all processes wait
        if self.master_process:
            self.print0(f'Saving checkpoint at step {step}...')

            if self.ddp_world_size > 1:
                model = self.model.module
            else:
                model = self.model

            log = dict(step=step, model=model.state_dict(), optimizers=[opt.state_dict() for opt in self.optimizers])
            if self.args.hf_model_name:
                try:
                    model.push_to_hub(self.args.hf_model_name, subfolder='step%06d' % step)
                except Exception as e:
                    self.print0(e)
                    self.print0(f'Pushing failed, defaulting to local save')
                    torch.save(log, 'logs/state_step%06d.pt' % step)
            else:
                torch.save(log, 'logs/state_step%06d.pt' % step)
        
        # Synchronize after saving
        if self.ddp_world_size > 1:
            dist.barrier()

    def train_step(self, step):
        self.model.train()
        
        # Clear cache periodically to prevent memory fragmentation
        if step % self.args.clear_cache_every == 0:
            torch.cuda.empty_cache()
        
        # Accumulate losses for proper averaging
        accumulated_loss = 0.0
        
        for i in range(self.args.grad_accum):
            with contextlib.ExitStack() as stack:
                # Only sync gradients on last accumulation step
                if self.ddp_world_size > 1 and i < self.args.grad_accum - 1:
                    stack.enter_context(self.model.no_sync())
                input_ids, labels, mask_rate = self.train_loader.next_batch()
                loss = self.model(input_ids, labels, mask_rate, self.sliding_window_size) / self.args.grad_accum
                loss.backward()
                accumulated_loss += loss.item()  # Accumulate the scaled loss

        # momentum warmup for Muon
        if self.args.use_muon:
            frac = min(step/self.args.muon_momentum_warmup_steps, 1)
            for group in self.optimizers[-1].param_groups:
                group['momentum'] = (1 - frac) * 0.85 + frac * 0.95

        # Apply gradient clipping if specified
        if self.args.grad_clip > 0:
            if self.ddp_world_size > 1:
                # For DDP, use the module's parameters
                torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.args.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

        # step the optimizers and schedulers
        for opt, sched in zip(self.optimizers, self.lr_schedulers):
            opt.step()
            sched.step()

        # null the gradients
        self.model.zero_grad(set_to_none=True)
        
        # Return the total accumulated loss (already properly scaled)
        return accumulated_loss

    def train(self):
        self.init_training()

        train_losses = []

        ### BEGIN TRAINING LOOP ###
        self.print0("Beginning training loop...")
        
        # Synchronize before starting training
        if self.ddp_world_size > 1:
            dist.barrier()
        
        # Show progress only on master
        pbar = tqdm(range(self.args.num_steps + 1), desc='Training steps', disable=not self.master_process)
        
        try:
            for step in pbar:
                if step == 10: # ignore first 10 steps of timing because they are slower
                    self.train_timer.reset()
                    self.train_timer.start()
                timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

                frac_done = step / self.args.num_steps  # training progress
                self.sliding_window_size = self.sliding_window_size_scheduler(frac_done)

                # once in a while evaluate the validation dataset
                if self.args.eval_every > 0 and step % self.args.eval_every == 0:
                    val_loss, val_perplexity, val_tokens = self._run_eval_loader_timed(self.valid_loader, prefix='Validation')
                    training_time_sec = self.train_timer.get_time()
                    step_avg_ms = 1000 * training_time_sec / (timed_steps - 1) if timed_steps > 1 else 0
                    self.print0(f'step:{step}/{self.args.num_steps} step_avg:{step_avg_ms:.2f}ms')
                    self.log_wandb(
                        {'loss': val_loss, 'perplexity': val_perplexity, 'tokens': val_tokens, 'sliding_window_size': self.sliding_window_size},
                        prefix='val'
                    )

                # save checkpoint every `save_every` steps
                if self.args.save_every:
                    if step % self.args.save_every == 0:
                        self._save_checkpoint_timed(step)

                loss = self.train_step(step)
                train_losses.append(loss)

                # everything that follows now is just eval, diagnostics, prints, logging, etc.
                if step % 100 == 0:
                    train_time_sec = self.train_timer.get_time()
                    avg_loss = sum(train_losses) / len(train_losses)
                    
                    # Gather training loss across all processes for accurate logging
                    if self.ddp_world_size > 1:
                        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
                        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
                        avg_loss = avg_loss_tensor.item()
                    
                    self.print0(f'step:{step+1}/{self.args.num_steps} train_time:{train_time_sec:.0f} sec step_avg:{1000*train_time_sec/timed_steps:.2f}ms loss:{avg_loss:.4f}')
                    train_losses = []

                # Log training progress to wandb
                if self.master_process and self.args.wandb_token:
                    log_dict = {
                        "time_sec": train_time_sec,
                        "step_avg_ms": 1000*train_time_sec/timed_steps if timed_steps > 0 else 0,
                        "step": step,
                        "loss": avg_loss
                    }
                    self.log_wandb(log_dict, prefix='train')

            # Stop the timer and get final training time
            self.train_timer.pause()
            final_training_time_sec = self.train_timer.get_time()
            
            self.print0(f'peak memory consumption training: {torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024} GiB')
            self.print0(f'Train Time: {final_training_time_sec:.0f}s | Step Avg: {final_training_time_sec/timed_steps:.2f}s')
            self.print0(f'Total train time (min): {final_training_time_sec / 60:.2f}')
            self.print0(f'Total train time (hours): {final_training_time_sec / 3600:.2f}')
            # save the model to huggingface
            self._save_checkpoint_timed(self.args.num_steps)

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            set_seed(self.args.seed)

            test_loss, test_perplexity, test_tokens = self._run_eval_loader_timed(self.test_loader, prefix='Test')

            self.print0(f"peak memory consumption testing: {torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024} GiB")
        
            # Final wandb logging
            if self.master_process and self.args.wandb_token:
                log_dict = {
                    "test_loss": test_loss,
                    "test_perplexity": test_perplexity,
                    "test_tokens": test_tokens.item() if hasattr(test_tokens, "item") else test_tokens,
                    "final_train_time_sec": final_training_time_sec,
                    "final_step_avg_sec": final_training_time_sec/(timed_steps-1) if timed_steps > 1 else 0,
                    "peak_memory_training_gb": torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024,
                }
                self.log_wandb(log_dict, prefix='test')

            # Log final summary
            log_dict = {
                "val_loss": val_loss,
                "test_loss": test_loss,
                "test_perplexity": test_perplexity,
                "train_time_sec": final_training_time_sec,
                "step_avg_sec": final_training_time_sec/(timed_steps-1) if timed_steps > 1 else 0,
                "peak_memory_training_gb": torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024,
            }
            self.log_wandb(log_dict, prefix='final')
            
        except KeyboardInterrupt:
            self.print0("\nTraining interrupted by user!")
        except Exception as e:
            self.print0(f"\nTraining failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up resources
            if self.master_process and self.args.wandb_token:
                wandb.finish()
        
            # clean up nice
            if self.ddp_world_size > 1:
                dist.destroy_process_group()


if __name__ == '__main__':
    args = arg_parser()

    if args.bugfix:
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

    model_config = PLMConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        num_att_tokens=args.num_att_tokens,
        vocab_size=args.vocab_size,
        expansion_ratio=args.expansion_ratio,
        soft_logit_cap=args.soft_logit_cap,
        attention_soft_cap=args.attention_soft_cap,
        add_att_soft_cap=args.add_att_soft_cap,
        p_attention=args.p_attention,
        tie_embeddings=args.tie_embeddings,
        unet=args.unet,
        mlm=args.mlm,
    )

    if args.token:
        from huggingface_hub import login
        login(args.token)
        # Clear token for security
        args.token = None 
        args.wandb_token = None
    
    trainer = Trainer(args, model_config)
    trainer.train()
