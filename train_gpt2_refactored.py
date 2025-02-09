import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import random
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import json
from dataclasses import dataclass
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast # type: ignore #
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.geoopt.optim import RiemannianSGD
from modules.model import GPT
from modules.muon import Muon
from modules.loader import DistributedDataLoader

import argparse

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FullConfig:
    # Data hyperparams
    data_path: str = "data/fineweb10B"
    input_bin: str = ""
    input_val_bin: str = ""
    num_vocab: int = 50304
    sequence_length: int = 1024
    # Optimization hyperparams
    batch_size: int = 64     # global batch size (across devices)
    device_batch_size: int = 32  # per-device batch size
    num_iterations: int = 1000
    cooldown_frac: float = 0.4
    weight_decay: float = 0
    # Evaluation/logging
    generate_every: int = 0
    train_loss_every: int = 10
    val_loss_every: int = 10
    val_tokens: int = 10_485_760
    save_every: int = 0
    # Model architecture
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    # Rather than specify n_embd directly,
    # you could also define a `head_dim`, if you like:
    head_dim: int = 128
    n_embd: int = 768
    head_mode: str = "euc"
    attn_mode: str = "euc"
    curvature: float = 1.0
    k_lr: float = 0.0
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """
        Dynamically set up paths and possibly recalculate n_embd from n_head, head_dim.
        You can also unify any validation logic here.
        """
        # If you want n_embd to be set from n_head * head_dim:
        self.n_embd = self.n_head * self.head_dim

        # Decide how to set the input bins.
        if "tinystories" in self.data_path:
            self.input_bin = f"{self.data_path}/train.bin"
            self.input_val_bin = f"{self.data_path}/val.bin"
        elif "fineweb" in self.data_path:
            self.input_bin = f"{self.data_path}/fineweb_train_*.bin"
            self.input_val_bin = f"{self.data_path}/fineweb_val_*.bin"
        else:
            raise ValueError("Specify a proper data path (contains 'tinystories' or 'fineweb')")


def compute_grad_norm(params):
    """Compute the total L2 norm of gradients in the given list of parameters."""
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)  # L2 norm
            total_norm_sq += param_norm.item() ** 2
    return total_norm_sq ** 0.5

parser = argparse.ArgumentParser(description="Train GPT model with customizable parameters.")

# Configurable arguments
parser.add_argument("--data_path", type=str, default="data/fineweb10B")
parser.add_argument("--num_iterations", type=int, default=4578)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--head_mode", type=str, default="euc", help="Set the mode for LM Head")
parser.add_argument("--attn_mode", type=str, default="euc", help="Set the mode for attention layers")
parser.add_argument("--curvature", type=float, default=1.0)
parser.add_argument("--k_lr", type=float, default=0.0)
parser.add_argument("--head_dim", type=int, default=128)
parser.add_argument("--n_head", type=int, default=6)

args = parser.parse_args()

# Create the config
config = FullConfig(
    data_path=args.data_path,
    num_iterations=args.num_iterations,
    seed=args.seed,
    head_mode=args.head_mode,
    attn_mode=args.attn_mode,
    curvature=args.curvature,
    k_lr=args.k_lr,
    head_dim=args.head_dim,
    n_head=args.n_head
)

# Seeds
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# Tokenizer setup
if "tinystories" in config.data_path:
    dataset_name = "TinyStories"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(config.data_path, "tinystories_tokenizer.json"),
        eos_token="<|endoftext|>",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
else:
    dataset_name = "FineWeb"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

# DDP setup
assert torch.cuda.is_available(), "CUDA is required for DDP but not available."

try:
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
except KeyError as e:
    raise RuntimeError(f"Missing environment variable for DDP: {e}")

# Initialize the process group
dist.init_process_group(backend='nccl')

# Map local rank to CUDA device
device = torch.device(f'cuda:{ddp_local_rank}')
torch.cuda.set_device(device)

print(f"[Rank {ddp_rank}] Using device: {device}")

# Identify master process
master_process = (ddp_rank == 0)
if master_process:
    print(f"[Rank {ddp_rank}] This is the master process.")

# Convenience variables
B, T = config.device_batch_size, config.sequence_length
assert config.val_tokens % (B * T * ddp_world_size) == 0, "val_tokens must be divisible by total global batch size."
val_steps = config.val_tokens // (B * T * ddp_world_size)

assert config.batch_size % (B * ddp_world_size) == 0, "batch_size must be divisible by global batch size."
train_accumulation_steps = config.batch_size // (B * ddp_world_size)

train_loader = DistributedDataLoader(config.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(config.input_val_bin, B, T, ddp_rank, ddp_world_size)

if master_process:
    print(f"[Rank {ddp_rank}] Training DataLoader: {train_loader.ntok_total} tokens across {len(train_loader.files)} files.")
    print(f"[Rank {ddp_rank}] Validation DataLoader: {val_loader.ntok_total} tokens across {len(val_loader.files)} files.")
x, y = train_loader.next_batch()

# Model setup
model = GPT(config, tokenizer)  # Step 1: Create the model on CPU
model = model.to(device)        # Step 2: Move the model to the correct device

# If using PyTorch 2.0+ and want compiled model:
model = torch.compile(model)

# Step 3: Wrap the model in DDP
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module  # Always access raw model via .module

# Optional: Verify that DDP is correctly set up
if master_process:
    print(f"[Rank {ddp_rank}] Model wrapped in DDP.")

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

head_k_params, attn_k_params = [], []
for name, param in raw_model.named_parameters():
    if "manifold.k" in name:
        head_k_params.append(param)
    elif "attn.k" in name:
        attn_k_params.append(param)

k_params = head_k_params + attn_k_params

if config.k_lr:
    for p in k_params:
        p.requires_grad = True
else:
    for p in k_params:
        p.requires_grad = False


if master_process:
    print(f"k params lengths: head = {len(head_k_params)}, attn = {len(attn_k_params)}")
        
lm_head_params = [p for name, p in raw_model.lm_head.named_parameters() if (p.requires_grad and ("manifold.k" not in name))]

params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
wte_params = [raw_model.transformer.wte.weight]

optimizer_lm_head = RiemannianSGD(
    [{'params': lm_head_params}], lr=0.1, weight_decay=5e-4, momentum=0.9, nesterov=True, stabilize=1
)

optimizer_muon = Muon(matrix_params, lr=0.05, momentum=0.95)

optimizer_wte = torch.optim.Adam(wte_params, lr=0.6, betas=(0.8, 0.95), fused=True)

if attn_k_params:
    optimizer_k = torch.optim.SGD([
        {"params": head_k_params, "lr": config.k_lr},  
        {"params": attn_k_params, "lr": config.k_lr}  
    ], momentum=0.9, nesterov=True)
    optimizers = [optimizer_lm_head, optimizer_muon, optimizer_wte, optimizer_k]
    if master_process:
        print(f"attn.k is learned")
elif head_k_params:
    optimizer_k = torch.optim.SGD([
        {"params": head_k_params, "lr": config.k_lr}
    ], momentum=0.9, nesterov=True)
    optimizers = [optimizer_lm_head, optimizer_muon, optimizer_wte, optimizer_k]
    if master_process:
        print(f"head.k is learned with {config.k_lr} lr")
else:
    optimizers = [optimizer_lm_head, optimizer_muon, optimizer_wte]
    if master_process:
        print(f"k is not learned")

init_lr = 1.0
end_lr  = 0.1
def get_lr(it):
    t = max(0, min(1, 1 - it/config.num_iterations))
    w = min(t / config.cooldown_frac, 1.0)
    return w * init_lr + (1 - w) * end_lr
    
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# schedulers[-1] = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizers[-1],
#     mode='max',           # Monitor for the minimum metric (e.g., validation loss).
#     factor=0.1,           # Reduce LR by a factor of 0.1.
#     patience=5,           # Wait for 5 epochs without improvement.
#     verbose=True          # Print LR reduction messages.
# )

if master_process:

    print("\n=== Minimal Report ===")
    print(f"Model Size:    {raw_model.model_size()}")

    # 3) Minimal set of relevant hyperparams
    print("\n=== Relevant Hyperparameters ===")
    print(f"Data Path:            {config.data_path}")
    print(f"Sequence Length:      {config.sequence_length}")
    print(f"Batch Size (global):  {config.batch_size}")
    print(f"Batch Size (device):  {config.device_batch_size}")
    print(f"n_layer:              {config.n_layer}")
    print(f"n_head:               {config.n_head}")
    print(f"head_dim:             {config.head_dim}")
    print(f"n_embd:               {config.n_embd}")
    print(f"Seed:                 {config.seed}")
    print("==============================\n")


# begin logging
if master_process:

    now = datetime.datetime.now()
    date_part = now.strftime('%d.%m')  
    seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).seconds #int(time.time() % 86400)
    run_id = f"{date_part}_{seconds_since_midnight}"

    if config.head_mode == 'hyp' and config.k_lr:
        suffix = f"k_{config.curvature}_lr_{config.k_lr:.1f}"
    elif config.head_mode == 'hyp':
        suffix = f"k_{config.curvature}"
    else:
        suffix = f"{config.head_mode}_head"

    # Construct the new folder name
    run_id = f"{run_id}_{suffix}_{config.seed}"
    
    # Create log directory and file
    logdir = f'runs/{run_id}/'
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "tensorboard_logs"), exist_ok=True)

    print(f"Logs for this run will be stored in: {logdir}")

    print("Writing logs to: " + os.path.join(logdir, "tensorboard_logs"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, "tensorboard_logs"))

    config_path = os.path.join(logdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    def pretty_json(hp):
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    writer.add_text("run_params", pretty_json(vars(args)))
    logfile = os.path.join(logdir, 'log.txt')
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_s = 0.0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
total_t0 = time.time()
train_loss_accum = 0.0
train_loss_count = 0
# begin training
train_loader.reset()
for step in range(config.num_iterations + 1):
    last_step = (step == config.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_s = 0.0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (config.val_loss_every > 0 and step % config.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_s += time.time() - t0
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            tokens_seen = step * config.batch_size * config.sequence_length
            print(f'step:{step}/{config.num_iterations}, tokens seen: {tokens_seen/1e6:.2f}M, val_loss:{val_loss:.4f} train_time:{training_time_s:.2f}s step_avg:{1000*training_time_s/(timed_steps-1):.0f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{config.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_s:.2f}s step_avg:{1000*training_time_s/(timed_steps-1):.0f}ms\n')
            writer.add_scalar('Loss/Validation', val_loss.item(), tokens_seen)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if config.generate_every and master_process and step and (step % config.generate_every == 0):
        # Use a fixed prompt or context for generation
        prompt = "Once upon a time"  # Customize as per your dataset
        context = raw_model.encode_text(prompt)
        
        # Generate text
        generated_tokens = raw_model.generate_text(context, max_length=200, temperature=1.0, top_k=50)
        generated_text = raw_model.decode_tokens(generated_tokens[0])
        
        # Log the generated text to TensorBoard
        writer.add_text(f"Generated_Text/Step_{step}", generated_text, step)
        
        # Optionally log to console for immediate feedback
        print(f"[Step {step}] Generated Text: {generated_text}")


    if master_process and (last_step or (config.save_every > 0 and step % config.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_s += time.time() - t0
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'ckpts/%s_state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step

    for name, p in model.named_parameters():
        if p.grad is None:
            # print(f"WARNING: Parameter {name} has no gradient. Skipping.")
            continue
        p.grad /= train_accumulation_steps

    if master_process and (step+1) % config.train_loss_every == 0:  # if you only want the rank-0 to log

        grad_norm_lm_head = compute_grad_norm(lm_head_params)
        grad_norm_matrix = compute_grad_norm(matrix_params)
        grad_norm_wte = compute_grad_norm(wte_params)
        grad_norm_k = compute_grad_norm(k_params)

        writer.add_scalar("grad_norm/lm_head", grad_norm_lm_head, step)
        writer.add_scalar("grad_norm/matrix", grad_norm_matrix, step)
        writer.add_scalar("grad_norm/wte", grad_norm_wte, step)
        writer.add_scalar("grad_norm/k", grad_norm_k, step)

    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    train_loss_accum += train_loss.item()
    train_loss_count += 1
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process and (step+1) % config.train_loss_every == 0:# within the main training loop, after logging validation loss or training loss
        
        if k_params:  # Only log if curvature is learnable
            for i, param in enumerate(head_k_params):
                curvature_value = param.item()  
                # curvature_grad = param.grad
                if i == 0:
                    print(f"Head curvature value: {curvature_value:.2f}")
                    writer.add_scalar(f"Curvature/Head", curvature_value, step)
            for i, param in enumerate(attn_k_params):
                curvature_value = param.item()  
                print(f"Attn curvature {i}: {curvature_value}")
                writer.add_scalar(f"Curvature/Attn/{i}", curvature_value, step)

        avg_train_loss = train_loss_accum / train_loss_count
        elapsed_time = time.time() - total_t0
        approx_time = training_time_s + (time.time() - t0)
        avg_time_per_step = approx_time/timed_steps
        estimated_total_time = avg_time_per_step * config.num_iterations
        tokens_seen = step * config.batch_size * config.sequence_length 
        print(f"step:{step+1}/{config.num_iterations}, tokens seen:{tokens_seen/1e6:.2f}M, avg_train_loss:{avg_train_loss:.4f} time:{elapsed_time:.0f}/{estimated_total_time:.0f}s step_avg:{1000*avg_time_per_step:.0f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{config.num_iterations} avg_train_loss:{avg_train_loss:.4f} time:{elapsed_time:.0f}s step_avg:{1000*avg_time_per_step:.0f}ms\n")
        writer.add_scalar('Loss/Train', avg_train_loss, tokens_seen)
        train_loss_accum = 0.0
        train_loss_count = 0

if master_process:
    total_training_time = time.time() - total_t0
    print(f"Total training time: {total_training_time:.2f}s")
    with open(logfile, "a") as f:
        f.write(f"Total training time: {total_training_time:.2f}s\n")
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
if master_process:
    writer.close()
dist.destroy_process_group()
