import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import math
import tyro
import wandb
from dataclasses import dataclass
import dataclasses

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class Scaler(nn.Module):

    def __init__(self, dim, init, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * scale)
        self.forward_scale = init / scale

    def forward(self):
        return self.scale * self.forward_scale

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # q,k,v projection
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_q.NORMALIZE = 1
        self.c_k.NORMALIZE = 1
        self.c_v.NORMALIZE = 1
        self.c_q.WEIGHT_HIDDEN = 1 # muP
        self.c_k.WEIGHT_HIDDEN = 1
        self.c_v.WEIGHT_HIDDEN = 1
        # q,k scaling
        self.qk_scaler = Scaler(dim=self.head_dim, init=1, scale=1/math.sqrt(config.n_embd))
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.RESIDUAL_SCALE_FLAG = 1
        self.c_proj.NORMALIZE = 1
        self.c_proj.NORM_FIRST = 1
        self.c_proj.WEIGHT_HIDDEN = 1
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        q = self.qk_scaler() * F.normalize(q, dim=-1) # nGPT step 4
        k = self.qk_scaler() * F.normalize(k, dim=-1)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=math.sqrt(self.head_dim)) # nGPT step 4
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_ff = int((8/3) * config.n_embd)
        self.n_embd = config.n_embd

        self.c_fc = nn.Linear(config.n_embd, d_ff, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, d_ff, bias=False)
        self.c_fc.NORMALIZE = 1
        self.c_fc2.NORMALIZE = 1
        self.c_fc.WEIGHT_HIDDEN = 1
        self.c_fc2.WEIGHT_HIDDEN = 1

        self.u_scaler = Scaler(dim=d_ff, init=1, scale=1)
        self.v_scaler = Scaler(dim=d_ff, init=1, scale=1)

        self.c_proj = nn.Linear(d_ff, config.n_embd, bias=False)
        self.c_proj.RESIDUAL_SCALE_FLAG = 1
        self.c_proj.NORMALIZE = 1
        self.c_proj.NORM_FIRST = 1
        self.c_proj.WEIGHT_HIDDEN = 1

    def forward(self, x):
        x1 = self.u_scaler() * self.c_fc(x) # nGPT step 5
        x2 = math.sqrt(self.n_embd) * self.v_scaler() * self.c_fc2(x) # nGPT step 5
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.attn_scaler = Scaler(dim=config.n_embd, init=1/config.n_layer, scale=1/math.sqrt(config.n_embd))
        self.mlp = MLP(config)
        self.mlp_scaler = Scaler(dim=config.n_embd, init=1/config.n_layer, scale=1/math.sqrt(config.n_embd))

    def forward(self, x):
        hA = F.normalize(self.attn(x), dim=-1) # nGPT step 3
        x = F.normalize(x + torch.abs(self.attn_scaler()) * (hA - x), dim=-1)
        hM = F.normalize(self.mlp(x), dim=-1)
        x = F.normalize(x + torch.abs(self.mlp_scaler()) * (hM - x), dim=-1)
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768
    mup_width_mult : float = 1

class GPT(nn.Module):

    def __init__(self, config, seed=None):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.transformer.wte.NORMALIZE = 1
        self.transformer.wte.WEIGHT_INPUT = 1

        self.logits_scaler = Scaler(dim=config.vocab_size, init=1, scale=1/math.sqrt(config.n_embd))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.NORMALIZE = 1
        self.lm_head.WEIGHT_OUTPUT = 1
        #self.lm_head.SKIP_INIT = 1 # don't init this one, we will tie weights
        #self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(seed or 42)
        if config.mup_width_mult != 1:
            self.apply(self._init_weights_mup)
        else:
            self.apply(self._init_weights)

        self.norm_weights()

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        
        if self.config.mup_width_mult != 1:
            x = x / self.config.mup_width_mult

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.logits_scaler() * self.lm_head(x) # nGPT step 6
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.logits_scaler() * self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            base_std = 1 / math.sqrt(self.config.n_embd)
            std = base_std if not hasattr(module, 'RESIDUAL_SCALE_FLAG') else base_std/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def _init_weights_mup(self, module):
        base_std = 0.02
        std = base_std if not hasattr(module, 'RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
        if hasattr(module, 'WEIGHT_INPUT'):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
        elif hasattr(module, 'WEIGHT_HIDDEN'):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult), generator=self.init_rng)
        elif hasattr(module, 'WEIGHT_OUTPUT'):
            torch.nn.init.zeros_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def configure_optimizer(self, learning_rate, weight_decay, betas):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, fused=True)
        return optimizer

    def configure_optimizer_mup(self, learning_rate, weight_decay, betas):
        params_INPUT_OUTPUT = []
        params_HIDDEN = []
        params_bias = []

        for module in self.modules():
            if hasattr(module, 'WEIGHT_INPUT') or hasattr(module, 'WEIGHT_OUTPUT'):
                params_INPUT_OUTPUT.append(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    params_bias.append(module.bias)
            elif hasattr(module, 'WEIGHT_HIDDEN'):
                params_HIDDEN.append(module.weight)
                if module.bias is not None:
                    params_bias.append(module.bias)
            elif isinstance(module, Scaler):
                params_INPUT_OUTPUT.append(module.scale)

        all_params = set(p for p in self.parameters() if p.requires_grad)
        params_in_groups = set(params_INPUT_OUTPUT + params_HIDDEN + params_bias)
        params_remaining = all_params - params_in_groups

        if params_remaining:
            for p in params_remaining:
                print(p.shape)
            print("some parameters are remaining")
            raise NotImplementedError

        param_groups = [
            {
                'params': params_INPUT_OUTPUT,
                'lr': learning_rate,
                'weight_decay': weight_decay
            },
            {
                'params': params_HIDDEN,
                'lr': learning_rate / self.config.mup_width_mult,
                'weight_decay': weight_decay * self.config.mup_width_mult
            },
            {
                'params': params_bias,
                'lr': learning_rate,
                'weight_decay': 0.
            }
        ]

        optimizer = torch.optim.AdamW(param_groups, betas=betas, fused=True)
        return optimizer

    @torch.no_grad()
    def norm_weights(self):
        for module in self.modules():
            if hasattr(module, 'NORMALIZE'):
                if hasattr(module, 'NORM_FIRST'): # W_o of SA and MLP
                    module.weight.copy_(F.normalize(module.weight, dim=0))
                else:
                    module.weight.copy_(F.normalize(module.weight, dim=-1))

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    seed : int = 123456789 + 0
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 16 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 4768 # number of iterations to run
    learning_rate : float = 2**(-9)
    warmup_iters : int = 0 # nGPT step 7
    warmdown_iters : int = 1450 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    grad_norm_clip : float = 1
    # mup
    use_mup : bool = False
    mup_base_width : int = 768
    # evaluation and logging hyperparams
    log_wandb: bool = True
    log_wandb_every: int = 12
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 2000 # every how many steps to save the checkpoint? 0 for only at the end

if __name__ == "__main__":
    args = tyro.cli(Hyperparameters)

    # set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

    if master_process and args.log_wandb:
        wandb.init(project="modded_gpt", config={**vars(args)})

    # convenience variables
    B, T = args.device_batch_size, args.sequence_length
    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = args.batch_size // (B * ddp_world_size)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    if master_process:
        print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    x, y = train_loader.next_batch()

    # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
    # this originates from Karpathy's experiments.
    num_vocab = 50304
    gptconfig = GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768)
    if args.use_mup:
        gptconfig.mup_width_mult = gptconfig.n_embd / args.mup_base_width
    model = GPT(gptconfig, seed=args.seed)
    model = model.cuda()
    if master_process:
        print(f"Model initialized. Number of parameters : {sum([p.numel() for p in model.parameters()])}.")
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    model = torch.compile(model)
    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # init the optimizer(s)
    if not args.use_mup:
        optimizer = raw_model.configure_optimizer(learning_rate=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:
        optimizer = raw_model.configure_optimizer_mup(learning_rate=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return (it+1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return decay_ratio
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # begin logging
    if master_process:
        if args.log_wandb:
            run_id = wandb.run.name
        else:
            run_id = str(uuid.uuid4())
        logdir = 'logs/%s/' % run_id
        os.makedirs(logdir, exist_ok=True)
        logfile = 'logs/%s.txt' % run_id
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
            # log hyperparameters and config
            if args.log_wandb:
                f.write(f"wandb run {wandb.run.name}\n")
            f.write("Hyperparameters:\n")
            f.write(f"{dataclasses.asdict(args)}\n")
            f.write("Model config:\n")
            f.write(f"{dataclasses.asdict(gptconfig)}\n")
            f.write('='*100 + '\n')

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.time()
    # begin training
    train_loader.reset()
    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        if step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        # once in a while evaluate the validation dataset
        if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
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
            # log val loss to console, logfile and wandb
            if master_process:
                print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
                with open(logfile, "a") as f:
                    f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
                if args.log_wandb:
                    wandb.log({"val_loss": val_loss}, step=step)
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # save the state of the training process
            log = dict(step=step, code=code, model=raw_model.state_dict(), optimizer=optimizer.state_dict())
            torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
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
        for p in model.parameters():
            p.grad /= train_accumulation_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_norm_clip, foreach=True)

        # step the optimizers and schedulers
        optimizer.step()
        scheduler.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        raw_model.norm_weights() # nGPT step 2

        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
            with open(logfile, "a") as f:
                f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
            if args.log_wandb and (step % args.log_wandb_every == 0):
                wandb.log({"train_loss": train_loss.item(), "lr": optimizer.param_groups[0]['lr'], "grad_norm": grad_norm.item(), "step_avg_ms": approx_time/timed_steps}, step=step)

    if master_process:
        print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    dist.destroy_process_group()
