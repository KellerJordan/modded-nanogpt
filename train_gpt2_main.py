import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import random
import datetime
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import json
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast # type: ignore #
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# from lib.geoopt.optim import RiemannianSGD

from model.model import GPT
from utils.muon import Muon
from utils.loader import DistributedDataLoader
from utils.config import FullConfig
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()

# Configurable arguments
parser.add_argument("--data_path", type=str, default="data/fineweb10B")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--device_batch_size", type=int, default=32)
parser.add_argument("--num_iterations", type=int, default=4578)
parser.add_argument("--generate_every", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--head_mode", type=str, default="euc", help="Set the mode for LM Head")
parser.add_argument("--attn_mode", type=str, default="euc", help="Set the mode for attention layers")
parser.add_argument("--curvature", type=float, default=1.0)
parser.add_argument("--k_lr", type=float, default=0.0)
parser.add_argument("--head_dim", type=int, default=128)
parser.add_argument("--n_heads", type=int, default=6)
parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer layers")
parser.add_argument("--sequence_length", type=int, default=1024)

args = parser.parse_args()

# Create the config
config = FullConfig(
    data_path=args.data_path,
    batch_size=args.batch_size,
    device_batch_size=args.device_batch_size,
    num_iterations=args.num_iterations,
    generate_every=args.generate_every,
    seed=args.seed,
    head_mode=args.head_mode,
    attn_mode=args.attn_mode,
    curvature=args.curvature,
    k_lr=args.k_lr,
    head_dim=args.head_dim,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    sequence_length=args.sequence_length
)

# Seeds
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# Tokenizer setup
if "shakespeare" in config.data_path:
    from data.shakespeare_char.CharTokenizer import CharacterTokenizer
    dataset_name = "TinyShakespeare"
    tokenizer = CharacterTokenizer.from_pretrained(save_directory="data/shakespeare_char/")
    config.vocab_size = tokenizer.vocab_size
elif "tinystories_char" in config.data_path:
    dataset_name = "TinyStoriesChar"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tinystories_char/char_tokenizer.json")
    config.vocab_size = tokenizer.vocab_size
elif "tinystories" in config.data_path:
    dataset_name = "TinyStories"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(config.data_path, "tinystories_tokenizer.json"),
        eos_token="<|endoftext|>",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    config.vocab_size = tokenizer.vocab_size
elif "fineweb" in config.data_path:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    if "finewebedu" in config.data_path:
        dataset_name = "FineWebEdu"
    else:
        dataset_name = "FineWeb"
else:
    raise ValueError("Incorrect data_path")

def encode_text(tokenizer, text, device):
    """Encodes a string into token IDs."""
    return tokenizer.encode(text, return_tensors="pt").to(device)

def decode_tokens(tokenizer, tokens):
    """Decodes token IDs into a readable string."""
    # For character-level tokenizer, join characters without spaces
    if "tinystories_char" in config.data_path:
        return ''.join(tokenizer.convert_ids_to_tokens(tokens.cpu().tolist()))
    # For word-level tokenizers, use normal decoding
    return tokenizer.decode(tokens.cpu().tolist(), skip_special_tokens=True)

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
    print(f"Training DataLoader: {train_loader.ntok_total / 1e6:.2f}M tokens across {len(train_loader.files)} files.")
    print(f"Validation DataLoader: {val_loader.ntok_total / 1e6:.2f}M tokens across {len(val_loader.files)} files.")
    print(f"config.val_tokens / val_loader.ntok_total = {config.val_tokens / val_loader.ntok_total:.2f}")
x, y = train_loader.next_batch()

# Model setup
model = GPT(config)  
model = model.to(device)    
# model = torch.compile(model)

# Step 3: Wrap the model in DDP
model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
raw_model = model.module  # Always access raw model via .module

# Optional: Verify that DDP is correctly set up
if master_process:
    print(f"[Rank {ddp_rank}] Model wrapped in DDP.")

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

head_k_params, attn_k_params = [], []
for name, param in raw_model.named_parameters():
    if "lm_head.k" in name:
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
    print(f"Tokenizer vocab size: {config.vocab_size}")

        
lm_head_params = [p for name, p in raw_model.lm_head.named_parameters() if (p.requires_grad and ("lm_head.k" not in name))]

params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
wte_params = [raw_model.transformer.wte.weight]

optimizer_head = torch.optim.Adam(lm_head_params, lr=0.22, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer_wte = torch.optim.Adam(wte_params, lr=0.6, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer_muon = Muon(matrix_params, lr=0.05, momentum=0.95)


if attn_k_params:
    optimizer_k = torch.optim.SGD([
        {"params": head_k_params, "lr": config.k_lr},  
        {"params": attn_k_params, "lr": config.k_lr}  
    ], momentum=0.9, nesterov=True)
    optimizers = [optimizer_head, optimizer_muon, optimizer_wte, optimizer_k]
    if master_process:
        print(f"attn.k is learned")
elif head_k_params:
    optimizer_k = torch.optim.SGD([
        {"params": head_k_params, "lr": config.k_lr}
    ], momentum=0.9, nesterov=True)
    optimizers = [optimizer_head, optimizer_muon, optimizer_wte, optimizer_k]
    if master_process:
        print(f"head.k is learned with {config.k_lr} lr")
else:
    optimizers = [optimizer_head, optimizer_muon, optimizer_wte]
    if master_process:
        print(f"k is not learned")

init_lr = 1.0
end_lr  = 0.1
def get_lr(it):
    t = max(0, min(1, 1 - it/config.num_iterations))
    w = min(t / config.cooldown_frac, 1.0)
    return w * init_lr + (1 - w) * end_lr
    
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

if master_process:
    model_size = raw_model.model_size()
    print("\n=== Model ===")
    print(f"Model Size:    {model_size}\n")
    print(f"Data Path:            {config.data_path}")
    print(f"Sequence Length:      {config.sequence_length}")
    print(f"Batch Size (global):  {config.batch_size}")
    print(f"Batch Size (device):  {config.device_batch_size}")
    print(f"n_layers:              {config.n_layers}")
    print(f"n_heads:               {config.n_heads}")
    print(f"head_dim:             {config.head_dim}")
    print(f"n_embd:               {config.n_embd}")
    print("\n=== Experiment ===")
    print(f"Head mode:             {config.head_mode}")
    print(f"Attention mode:        {config.attn_mode}")
    print(f"Init curvature:        {config.curvature}")
    print(f"Curvature learning rate: {config.k_lr}")
    print(f"Seed:                 {config.seed}")
    print("==============================\n")


# begin logging
if master_process:
    def create_run_id(config, dataset_name, timestamp):
        """Create a run identifier."""
        # Aliases for common configurations
        dataset_aliases = {
            'TinyShakespeare': 'sh',
            'TinyStoriesChar': 'tsc',
            'TinyStories': 'ts',
            'FineWeb': 'fw'
        }
        mode_aliases = {
            'euc': 'e',
            'hyp': 'h'
        }
        
        # Get date and time components
        date = timestamp.strftime('%m.%d') 
        seconds_since_midnight = (timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)).seconds
        
        # Get architecture configuration
        head, attn = mode_aliases[config.head_mode], mode_aliases[config.attn_mode]
        arch = f"{head}{attn}"
        
        # Build the hyperbolic parameters string if needed
        hyp_params = ""
        if 'h' in arch:
            hyp_params = f"_k{config.curvature}"
            if config.k_lr:
                hyp_params += f"_lr{config.k_lr:.0e}"  # Using shorter scientific notation
        
        # Combine all components
        run_id = f"{seconds_since_midnight}_{dataset_aliases[dataset_name]}_{arch}{hyp_params}_s{config.seed}"
        return date, run_id

    # Create the run ID
    now = datetime.datetime.now()
    date, run_id = create_run_id(config, dataset_name, now)
    # Create log directory and file
    logdir = f'runs/{date}/{run_id}/'
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


def compute_grad_norm(params):
    """Compute the total L2 norm of gradients in the given list of parameters."""
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)  # L2 norm
            total_norm_sq += param_norm.item() ** 2
    return total_norm_sq ** 0.5

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

    if config.generate_every and master_process and ((step) % config.generate_every == 0):
        # Use a fixed prompt or context for generation
        prompt = "Once upon a time in a"  # Customize as per your dataset
        context = encode_text(tokenizer,prompt, device)
        
        # Generate text
        generated_tokens = raw_model.generate_text(context, max_length=50, temperature=1.0, top_k=50)
        generated_text = decode_tokens(tokenizer, generated_tokens[0])
        
        # Log the generated text to TensorBoard
        writer.add_text(f"Generated_Text/Step_{step}", generated_text, step)
        
        # Optionally log to console for immediate feedback
        print(f"[Step {step}] Generated Text: {generated_text}")

        # Add curvature logging here
        if k_params:  # Only log if curvature is learnable
            # Log head curvature
            for i, param in enumerate(head_k_params):
                curvature_value = param.item()  
                if i == 0:
                    print(f"Head curvature value: {curvature_value:.2f}")
                    writer.add_scalar(f"Curvature/Head", curvature_value, step)
            
            # Log attention layer curvatures
            for i, param in enumerate(attn_k_params):
                curvature_values = param.squeeze().detach().cpu()  # Shape: (n_heads,)
                values_str = ' '.join([f"{v:.2f}" for v in curvature_values])
                print(f"Attn layer {i} curvatures: [{values_str}]")
                
                # Log each head's curvature to tensorboard
                for head_idx, value in enumerate(curvature_values):
                    writer.add_scalar(f"Curvature/Attn/{i}/Head_{head_idx}", value, step)

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

    if master_process and step % config.train_loss_every == 0:

        grad_norm_lm_head = compute_grad_norm(lm_head_params)
        grad_norm_matrix = compute_grad_norm(matrix_params)
        grad_norm_wte = compute_grad_norm(wte_params)
        grad_norm_head_k = compute_grad_norm(head_k_params)
        grad_norm_attn_k = compute_grad_norm(attn_k_params)

        writer.add_scalar("grad_norm/lm_head", grad_norm_lm_head, step)
        writer.add_scalar("grad_norm/matrix", grad_norm_matrix, step)
        writer.add_scalar("grad_norm/wte", grad_norm_wte, step)
        writer.add_scalar("grad_norm/head_k", grad_norm_head_k, step)
        writer.add_scalar("grad_norm/attn_k", grad_norm_attn_k, step)

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
    if master_process and step % config.train_loss_every == 0:# within the main training loop, after logging validation loss or training loss
        
        avg_train_loss = train_loss_accum / train_loss_count
        elapsed_time = time.time() - total_t0
        approx_time = training_time_s + (time.time() - t0)
        avg_time_per_step = approx_time/timed_steps
        estimated_total_time = avg_time_per_step * config.num_iterations
        tokens_seen = step * config.batch_size * config.sequence_length 
        print(f"step:{step}/{config.num_iterations}, tokens seen:{tokens_seen/1e6:.2f}M, avg_train_loss:{avg_train_loss:.4f} time:{elapsed_time:.0f}/{estimated_total_time:.0f}s step_avg:{1000*avg_time_per_step:.0f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step}/{config.num_iterations} avg_train_loss:{avg_train_loss:.4f} time:{elapsed_time:.0f}s step_avg:{1000*avg_time_per_step:.0f}ms\n")
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
