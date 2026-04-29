"""
train_gpt_ddp.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dion import NorMuon


os.environ["TORCHINDUCTOR_PERSISTENT_REDUCTIONS"] = "0"        
os.environ["TORCHINDUCTOR_MIX_ORDER_REDUCTION"] = "0"
########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

class StepProfiler:
    def __init__(self, start_step=20, end_step=25, trace_path="chrome_trace.json"):
        assert start_step <= end_step
        self.start_step = start_step
        self.end_step = end_step
        self.trace_path = trace_path
        self.prof = None
        self.active = False

    def maybe_start(self, step):
        if dist.get_rank() != 0 or step != self.start_step:
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
        )
        self.prof.__enter__()
        self.active = True
        print0(f"Started PyTorch profiler at step {step}; exporting to {self.trace_path}", console=True)

    def step_context(self, step):
        if self.active and self.start_step <= step <= self.end_step:
            return torch.profiler.record_function(f"train_step_{step}")
        return nullcontext()

    def maybe_stop(self, step):
        if not self.active or step != self.end_step:
            return
        assert self.prof is not None
        torch.cuda.synchronize()
        self.prof.__exit__(None, None, None)
        self.prof.export_chrome_trace(self.trace_path)
        self.active = False
        print0(f"Exported PyTorch profiler trace to {self.trace_path}", console=True)

A100_BF16_PEAK_FLOPS = 312e12

def estimate_flops_per_token(model, num_layers: int, num_heads: int, head_dim: int, seq_len: int):
    num_params = sum(p.numel() for p in model.parameters())
    return 6 * num_params + 12 * num_layers * num_heads * head_dim * seq_len

def estimate_a100_mfu(completed_steps: int, elapsed_seconds: float, flops_per_token: int):
    if completed_steps <= 0 or elapsed_seconds <= 0:
        return 0.0
    achieved_flops = completed_steps * batch_size * flops_per_token / elapsed_seconds
    peak_flops = dist.get_world_size() * A100_BF16_PEAK_FLOPS
    return 100 * achieved_flops / peak_flops

def estimate_a100_step_mfu(step_seconds: float, flops_per_token: int):
    if step_seconds <= 0:
        return 0.0
    achieved_flops = batch_size * flops_per_token / step_seconds
    peak_flops = dist.get_world_size() * A100_BF16_PEAK_FLOPS
    return 100 * achieved_flops / peak_flops

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running on device_name={torch.cuda.get_device_name(device)} with world_size={dist.get_world_size()}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

########################################
#       Init & Optim Hyperparams       #
########################################

# we want to minimize this while still reaching 3.28 val loss
train_steps = 3500

raw_model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
raw_model.compile(dynamic=False)
flops_per_token = estimate_flops_per_token(raw_model, num_layers=12, num_heads=6, head_dim=128, seq_len=1024)
print0(f"Estimated FLOPs/token:{flops_per_token} using A100 BF16 peak:{A100_BF16_PEAK_FLOPS:.3e} FLOP/s per GPU")

# initialize model parameters before DDP broadcasts them from rank 0
for name, p in raw_model.named_parameters():
    if "proj" in name:
        p.data.zero_()

model = DDP(raw_model, device_ids=[device.index], broadcast_buffers=False,
            gradient_as_bucket_view=True, bucket_cap_mb=128)

# create the optimizer(s)
optimizer1 = AdamW([dict(params=[raw_model.embed.weight], lr=0.3),
                    dict(params=[raw_model.proj.weight], lr=1/320),
                    dict(params=[p for p in raw_model.parameters() if p.ndim < 2], lr=0.01)],
                   betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
optimizer2 = NorMuon([p for p in raw_model.blocks.parameters() if p.ndim >= 2],
                     distributed_mesh=dist.group.WORLD,
                     lr=0.025, mu=0.95, muon_beta2=0.95, weight_decay=0.0125,
                     nesterov=True, adjust_lr="spectral_norm",
                     use_gram_newton_schulz=True)
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(raw_model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def set_hparams(step, cooldown_frac=0.7):
    progress = step / train_steps
    assert 0 <= progress < 1
    if progress < 1 - cooldown_frac:
        eta = 1.0
    else:
        eta = (1 - progress) / cooldown_frac
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta


########################################
#        Training and Validation       #
########################################

# start the clock
training_time = 0
last_training_time = 0
dist.barrier()
t0 = time.perf_counter()
profiler = StepProfiler(start_step=20, end_step=25, trace_path="chrome_trace.json")
for step in range(train_steps + 1):

    # --------------- VALIDATION SECTION -----------------
    if step == train_steps or step % 125 == 0:
        # stop the clock
        dist.barrier()
        training_time += time.perf_counter() - t0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += raw_model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        mfu = estimate_a100_mfu(step, training_time, flops_per_token)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*training_time/max(step, 1):.2f}ms mfu:{mfu:.2f}%", console=True)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if step == train_steps:
        break

    # --------------- TRAINING SECTION -----------------
    train_step = step + 1
    profiler.maybe_start(train_step)
    with profiler.step_context(train_step):
        inputs, targets = next(train_loader)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        num_microbatches = len(inputs) // mbs
        for i in range(num_microbatches):
            sync_context = nullcontext() if i == num_microbatches - 1 else model.no_sync()
            with sync_context:
                loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
                (loss * dist.get_world_size()).backward()
        for name, p in raw_model.named_parameters():
            assert p.grad is not None, name
        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        raw_model.zero_grad(set_to_none=True)
    profiler.maybe_stop(train_step)
    approx_training_time = training_time + (time.perf_counter() - t0)
    step_time = approx_training_time - last_training_time
    last_training_time = approx_training_time
    mfu = estimate_a100_step_mfu(step_time, flops_per_token)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
           + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms"
           + f" step_time:{1000*step_time:.2f}ms mfu:{mfu:.2f}%", console=True, log=False)

dist.destroy_process_group()
