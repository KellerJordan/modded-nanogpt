"""
K-Maxwell first-moment variant of PR #351's MuonH fast-slow-decay trainer.

The winning K=6 kernel uses log-spaced EMA ages in [3, 64] and linearly anneals
its convex mix from mean age 50 to 22 after a lazy-init switch at step 750.
Only MuonH's first moment changes; its NS direction, hyperball projection,
parameter groups, LR schedule, and the auxiliary AdamW are unchanged.

At the switch, all K buffers initialize from the just-advanced single-EMA
momentum, so that step is identical to the baseline. The submitted schedule is
the default; only --seed varies across the n=8 confirmation:

  torchrun --standalone --nproc_per_node=8 \
      records/track_3_optimization/results/20260827_muonh_kmaxwell_3065/train_gpt_muonh_kmaxwell.py \
      --seed 0
"""

import os
import sys
import argparse

with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import time
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

# Parse arguments (must handle torchrun's extra args)
parser = argparse.ArgumentParser()
parser.add_argument("--warmup_end", type=int, default=100,
                    help="End of the linear warmup phase (Phase 1). LR ramps 0 -> peak_lr on [0, warmup_end).")
parser.add_argument("--plateau_end", type=int, default=200,
                    help="End of an optional constant plateau at peak_lr (Phase 2). If set (>warmup_end), LR is held at peak_lr on [warmup_end, plateau_end), then the fast-decay phase runs on [plateau_end, fast_decay_end). Default = warmup_end (no plateau).")
parser.add_argument("--fast_decay_end", type=int, default=1750,
                    help="End of the fast-decay phase (Phase 3); tune together with floor_lr.")
parser.add_argument("--peak_lr", type=float, default=0.030,
                    help="MuonH LR at the top of the schedule (end of warmup / on the plateau).")
parser.add_argument("--floor_lr", type=float, default=0.006,
                    help="MuonH LR at the end of the fast-decay phase (start of the slow-decay phase).")
parser.add_argument("--min_lr", type=float, default=0.0,
                    help="MuonH LR at the very end of training (end of the slow-decay phase).")
parser.add_argument("--slow_decay_schedule", type=str, default="linear",
                    choices=["linear", "cosine", "minus_sqrt"],
                    help="Decay shape used in the slow-decay phase (Phase 4): floor_lr -> min_lr.")
parser.add_argument("--train_steps", type=int, default=3125, help="Total training steps")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (None = unseeded)")
parser.add_argument("--fast_decay_exponent", type=float, default=0.6,
                    help="Power exponent used in the fast-decay phase (peak_lr -> floor_lr). 1.0 = linear, <1.0 = concave (fast drop then slow), >1.0 = convex.")
args, _ = parser.parse_known_args()

WARMUP_END = args.warmup_end
PLATEAU_END = args.plateau_end if args.plateau_end is not None else args.warmup_end
FAST_DECAY_END = args.fast_decay_end
PEAK_LR = args.peak_lr
FLOOR_LR = args.floor_lr
MIN_LR = args.min_lr
SLOW_DECAY_SCHEDULE = args.slow_decay_schedule
TRAIN_STEPS = args.train_steps
SEED = args.seed
FAST_DECAY_EXPONENT = args.fast_decay_exponent

# Winning K-Maxwell recipe: K=6, log-spaced ages [3, 64], mean age 50 -> 22.
KMAXWELL_DECAY_RATES = [
    0.75, 0.8469227055704172, 0.9107413502519109,
    0.9495389483727874, 0.9719912250466763, 0.9846153846153846,
]
KMAXWELL_START_WEIGHTS = [
    0.021003991507788214, 0.04200798301557643, 0.06301197452336464,
    0.08401596603115286, 0.10501995753894107, 0.6849401273831768,
]
KMAXWELL_END_WEIGHTS = [
    0.06301197452336468, 0.12602394904672937, 0.18903592357009405,
    0.25204789809345873, 0.31505987261682344, 0.05482038214952978,
]
KM_START = 750
KM_ANNEAL_STEPS = TRAIN_STEPS - KM_START

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
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

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
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
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
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

@torch.compile
def muon_update_kmaxwell(grad, momenta, decay_rates, weights, mu=0.95):
    lerp_coeff = (1 - decay_rates).reshape(-1, *([1] * grad.ndim))
    momenta.lerp_(grad.unsqueeze(0), lerp_coeff)
    mix_shape = (-1, *([1] * grad.ndim))
    update = grad.lerp_((weights.reshape(mix_shape) * momenta).sum(dim=0), mu)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

def scale_invariant_update_(param: Tensor, update: Tensor, lr: float, eps: float = 1e-10) -> None:
    """Hyperball-constrained step: take a Muon-orthogonalised update of size lr * ||param||,
    then renormalise back onto the Frobenius sphere of the parameter's initial radius. Preserves
    ||param|| exactly across training; the invariant lets us drop weight decay on hidden
    matrices entirely (the constraint already prevents norm growth)."""
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    new_norm = torch.clamp(new_param.norm(), min=eps)
    param.copy_(new_param / new_norm * p_norm)

class MuonH(torch.optim.Optimizer):
    """MuonH: same Newton-Schulz orthogonalised direction as Muon, applied via a Frobenius-
    norm-preserving hyperball projection. Used here for ALL hidden 2D weight matrices —
    q, k, v, mlp.fc, attn.proj, mlp.proj — under non-zero (Kaiming-derived) init."""
    def __init__(self, params, lr=0.014, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)
        self._step = 0
        self.km_decay_rates = torch.tensor(KMAXWELL_DECAY_RATES)
        self.km_start_weights = torch.tensor(KMAXWELL_START_WEIGHTS)
        self.km_end_weights = torch.tensor(KMAXWELL_END_WEIGHTS)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    if self._step >= KM_START:
                        if "momenta" not in state:
                            update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                            state["momenta"] = state["momentum"].unsqueeze(0).repeat(
                                len(KMAXWELL_DECAY_RATES), *([1] * p.ndim))
                        else:
                            decay_rates = self.km_decay_rates.to(p)
                            frac = min(1.0, (self._step - KM_START) / KM_ANNEAL_STEPS)
                            weights = self.km_start_weights.lerp(
                                self.km_end_weights, frac).to(p)
                            update = muon_update_kmaxwell(
                                p.grad, state["momenta"], decay_rates, weights,
                                mu=group["mu"])
                    else:
                        update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    scale_invariant_update_(p, update, group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self._step += 1


########################################
#     Fast-Slow-Decay LR Schedule      #
########################################

def fast_slow_decay_lr(step: int,
                       max_iters: int,
                       warmup_end: int,
                       fast_decay_end: int,
                       peak_lr: float,
                       floor_lr: float,
                       min_lr: float,
                       slow_decay_schedule: str = "linear",
                       fast_decay_exponent: float = 1.0,
                       plateau_end: int = None) -> float:
    """Fast-slow-decay MuonH learning-rate schedule (four phases).

    Phase 1 Warmup      [0, warmup_end):
        linear from 0 to peak_lr
    Phase 2 Plateau     [warmup_end, plateau_end):
        hold at peak_lr (only if plateau_end > warmup_end)
    Phase 3 Fast decay  [plateau_end, fast_decay_end):
        peak_lr -> floor_lr along y = peak_lr + (floor_lr - peak_lr) * progress**fast_decay_exponent.
        With fast_decay_exponent < 1.0 this is *concave*: LR drops fast at the
        start of the phase, then flattens toward floor_lr -- the "fast" half of
        the fast-slow decay.
    Phase 4 Slow decay  [fast_decay_end, max_iters]:
        floor_lr -> min_lr along `slow_decay_schedule` (linear / cosine / minus_sqrt).
        Long and gentle -- the "slow" half.
    """
    if plateau_end is None or plateau_end < warmup_end:
        plateau_end = warmup_end
    if step < warmup_end:
        # Phase 1: linear warmup from 0 to peak_lr
        return peak_lr * (step + 1) / warmup_end
    elif step < plateau_end:
        # Phase 2: constant plateau at peak_lr
        return peak_lr
    elif step < fast_decay_end:
        # Phase 3: fast-decay (power) descent from peak_lr to floor_lr
        progress = (step - plateau_end) / (fast_decay_end - plateau_end)
        if fast_decay_exponent == 1.0:
            frac = progress
        elif progress <= 0.0:
            frac = 0.0
        else:
            frac = progress ** fast_decay_exponent
        return peak_lr + (floor_lr - peak_lr) * frac
    else:
        # Phase 4: slow-decay from floor_lr to min_lr
        total_slow = max_iters - fast_decay_end
        progress = (step - fast_decay_end) / total_slow  # 0 -> 1
        progress = min(max(progress, 0.0), 1.0)
        if slow_decay_schedule == "linear":
            coeff = 1.0 - progress
        elif slow_decay_schedule == "cosine":
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif slow_decay_schedule == "minus_sqrt":
            coeff = 1.0 - math.sqrt(progress)
        else:
            raise ValueError(f"Unknown slow_decay_schedule: {slow_decay_schedule}")
        return min_lr + coeff * (floor_lr - min_lr)


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

# Optional deterministic seeding (per-rank offset so ranks see different data / init dropout, but
# runs with the same --seed produce the same result end-to-end).
if SEED is not None:
    import random as _py_random
    _rank = dist.get_rank()
    _rank_seed = SEED + _rank
    _py_random.seed(_rank_seed)
    try:
        import numpy as _np
        _np.random.seed(_rank_seed)
    except Exception:
        pass
    torch.manual_seed(_rank_seed)
    torch.cuda.manual_seed_all(_rank_seed)

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    _seed_tag = f"_seed{SEED}" if SEED is not None else ""
    _steps_tag = f"_steps{TRAIN_STEPS}" if TRAIN_STEPS != 3100 else ""
    _exp_tag = f"_exp{FAST_DECAY_EXPONENT}" if FAST_DECAY_EXPONENT != 1.0 else ""
    _plateau_tag = f"_plateauEnd{PLATEAU_END}" if PLATEAU_END != WARMUP_END else ""
    logfile = (f"logs/muonh_kmaxwell_warm{WARMUP_END}{_plateau_tag}_peak{PEAK_LR}"
               f"_fastEnd{FAST_DECAY_END}_floor{FLOOR_LR}_slow{SLOW_DECAY_SCHEDULE}"
               f"{_exp_tag}{_steps_tag}{_seed_tag}.txt")
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0(f"Config: warmup_end={WARMUP_END}, plateau_end={PLATEAU_END}, peak_lr={PEAK_LR},"
       f" fast_decay_end={FAST_DECAY_END}, floor_lr={FLOOR_LR},"
       f" min_lr={MIN_LR}, slow_decay_schedule={SLOW_DECAY_SCHEDULE},"
       f" fast_decay_exponent={FAST_DECAY_EXPONENT},"
       f" train_steps={TRAIN_STEPS}, seed={SEED}")
print0(f"K-Maxwell: k=6, tau=[3,64], mean_age=50->22, start={KM_START}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)

########################################
#       Init & Optim Hyperparams       #
########################################

# we want to minimize this while still reaching 3.28 val loss
train_steps = TRAIN_STEPS
assert WARMUP_END <= PLATEAU_END < FAST_DECAY_END < train_steps, \
    f"Require warmup_end <= plateau_end < fast_decay_end < train_steps; got {WARMUP_END}, {PLATEAU_END}, {FAST_DECAY_END}, {train_steps}"

# initialize model parameters (same as #37)
for name, p in model.named_parameters():
    w = p.data
    if name.endswith("weight"):
        if "embed" in name:
            w.normal_()  # default torch init
        else:
            w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
    elif name.endswith("bias"):
        w.zero_()
    elif name.endswith("gains"):
        w.normal_(mean=1, std=0)
    else:
        raise Exception(f"Uninitialized parameter: {name}")
    if name.endswith(".attn.proj.weight"):
        w.mul_(1.25)
    elif name.endswith(".mlp.proj.weight"):
        w.mul_(3.0)
    elif name.endswith(".mlp.fc.weight"):
        w.mul_(1.5)

# create the optimizer(s)
# AdamW (aux) is FIXED at #37 baseline settings
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.910),
                    dict(params=[model.proj.weight], lr=0.0064),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.0195)],
                   betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
# MuonH: base lr set to 1.0 so the fast-slow-decay schedule returns absolute LRs.
muonh_params = [p for p in model.blocks.parameters() if p.ndim == 2]
optimizer2 = MuonH(muonh_params, lr=1.0)
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# LR schedule:
#   - AdamW (aux): FIXED at #37 baseline (cooldown_frac=0.85, linear decay)
#   - MuonH: four-phase fast-slow-decay schedule
#       Phase 1 Warmup      [0, warmup_end):        linear 0 -> peak_lr
#       Phase 2 Plateau     [warmup_end, plateau_end):  hold at peak_lr
#       Phase 3 Fast decay  [plateau_end, fast_decay_end): power decay peak_lr -> floor_lr
#       Phase 4 Slow decay  [fast_decay_end, train_steps]: slow_decay_schedule floor_lr -> min_lr
def set_hparams(step):
    progress = step / train_steps
    assert 0 <= progress < 1

    # --- AdamW schedule: fixed baseline (cooldown_frac=0.85) ---
    aux_cooldown_frac = 0.85
    for group in optimizer1.param_groups:
        if progress < 1 - aux_cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / aux_cooldown_frac
        group["lr"] = group["initial_lr"] * eta

    # --- MuonH schedule: four-phase fast-slow-decay ---
    muon_lr = fast_slow_decay_lr(step, train_steps,
                                 warmup_end=WARMUP_END,
                                 fast_decay_end=FAST_DECAY_END,
                                 peak_lr=PEAK_LR,
                                 floor_lr=FLOOR_LR,
                                 min_lr=MIN_LR,
                                 slow_decay_schedule=SLOW_DECAY_SCHEDULE,
                                 fast_decay_exponent=FAST_DECAY_EXPONENT,
                                 plateau_end=PLATEAU_END)
    for group in optimizer2.param_groups:
        # initial_lr is 1.0, so this is exactly muon_lr
        group["lr"] = group["initial_lr"] * muon_lr

########################################
#        Training and Validation       #
########################################

train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
last_val_step = 0
dist.barrier()
t0 = time.perf_counter()
for step in range(train_steps + 1):

    # --------------- VALIDATION SECTION -----------------
    dense = 3000 <= step <= train_steps and step % 5 == 0
    if step == train_steps or step % 125 == 0 or dense:
        # stop the clock
        dist.barrier()
        time_since_last_val = time.perf_counter() - t0
        step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
        last_val_step = step
        training_time += time_since_last_val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*step_avg:.2f}ms", console=True)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if step == train_steps:
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    for i in range(len(inputs) // mbs):
        model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    # set optimization hyperparameters and take a step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time = training_time + (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
           + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

dist.destroy_process_group()
