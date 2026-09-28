"""
Hi reader, sorry this method is still slop. It is more principled than k-maxwell from #357 and #359.
It is not yet a complete theory of momentum, nor would I necessarily recommend it yet outside nanogpt.
I'm PRing this since the gains appear to be legitimate and I want others in the community to be able
to build on top of these findings while I figure out why this works. -- @jeffreycider

MaxEntSlop first-moment variant of PR #351's MuonH fast-slow-decay trainer.

From step 750 the momentum direction is a weighted sum of the parameter's last
256 gradients (weight for age 0 = the current gradient, age 1 = the previous
one, ...). The 256 weights, a MomentumKernel, are solved at startup from a
MomentumDesiderata (mean gradient age, log-age moment, the two newest weights)
and anneal linearly from an early kernel (mean age 90 steps) to a late kernel
(mean age 24.05 steps) over the rest of training. The temporal weights and
radial adjustment change the input to Newton-Schulz. Its numerical calculation,
the sphere retraction, parameter groups and auxiliary AdamW are inherited from
the parent trainer. Learning-rate values are specified by the launch command
below.

The measured intervention also sets, before Newton-Schulz, the part of the
momentum sum parallel to the current parameter matrix. Its length is 0.044
times the norm of the weight-perpendicular part, and it keeps the sign of the
original weight-parallel part. No history age enters this rule. This placement
and ratio are measured; why they help remains open.

The gradient history is recorded from step 0. The measured tuning configuration
uses the following explicit clock and ending learning rate; the script defaults
do not reproduce it. Confirmation across eight consecutive seeds on this exact
standalone script remains required before a record claim.

  torchrun --standalone --nproc_per_node=8 \
      train_gpt_muonh_maxentslop_gamma.py \
      --train_steps 3045 --min_lr 0.000349 --seed 0
"""

import os
import sys
import argparse
from dataclasses import dataclass

import numpy as np

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

MAXENTSLOP_START = 750                      # first step that uses a momentum kernel; before it, ordinary Nesterov momentum
MAXENTSLOP_ANNEAL_STEPS = TRAIN_STEPS - MAXENTSLOP_START
MAXENTSLOP_HISTORY_LENGTH = 256             # how many past gradients each parameter keeps = the length of a momentum kernel
RADIAL_TO_TANGENT_RATIO = 0.044             # retained signed radial length / tangent Frobenius norm

# I call a "momentum kernel" the weights that we assign to lagged gradients.
# At lag t (t = 0 is the current gradient), EMA momentum has kernel weight (1 - beta) * beta**t.
# To calculate momentum, we just do np.dot(momentum_kernel, gradient_history).
MomentumKernel = torch.Tensor      # 1-D float32, length MAXENTSLOP_HISTORY_LENGTH, entry t = weight of the lag-t gradient

@dataclass(frozen=True)
class MomentumDesiderata:
    """We numerically solve for momentum kernels that have certain properties. Just a bookkeeping class to track the desiderata.

    mean_lag:             sum_k k * kernel[k], the mean age in steps of the gradients the direction averages over.
    log_moment:           sum_k log(1 + k) * kernel[k], the mean log-age; together with mean_lag it fixes how
                          the weight is spread between recent and old gradients.
    newest_weight:        kernel[0], the weight of the current gradient.
    second_newest_weight: kernel[1], the weight of the previous gradient.
    length:               number of kernel entries (ages 0 .. length-1).
    Always also imposed: the weights sum to 1, and sum_k (-1)**k * kernel[k] = 0, so a gradient that flips sign
    every step (a period-2 oscillation) contributes nothing to the direction.
    """
    mean_lag: float
    log_moment: float
    newest_weight: float
    second_newest_weight: float
    length: int = MAXENTSLOP_HISTORY_LENGTH

def solve_for_momentum_kernel_given_desiderata(d: MomentumDesiderata) -> MomentumKernel:
    """Maximum-entropy momentum kernel meeting the desiderata exactly.

    kernel[0] and kernel[1] are set to the two requested weights. Entries for ages k >= 2 maximise the entropy
    -sum(w log w) subject to the four equalities (sum, zero period-2 response, mean_lag, log_moment). The maximiser
    has the exponential-family form w[k] = exp(a + b*(-1)**k + c*k + d*log(1+k)); its four multipliers are found by
    Newton's method on the convex dual with a backtracking line search, to machine precision in a few dozen steps.
    """
    k = np.arange(2, d.length, dtype=float)
    feats = np.vstack([(-1.0) ** k, k, np.log1p(k)])                       # 3 x (length - 2)
    w0, w1 = d.newest_weight, d.second_newest_weight
    mass = 1.0 - w0 - w1                                                     # weight left for ages >= 2
    target = np.array([-(w0 - w1), d.mean_lag - w1, d.log_moment - w1 * np.log(2.0)]) / mass
    scale = np.ptp(feats, axis=1)                                            # condition the features
    F_, t = feats / scale[:, None], target / scale
    def dual(lam):
        z = F_.T @ lam
        return z.max() + np.log(np.exp(z - z.max()).sum()) - lam @ t
    lam = np.zeros(3)
    for _ in range(500):
        z = F_.T @ lam
        p = np.exp(z - z.max()); p /= p.sum()                                # tail weights, normalised to 1
        grad = F_ @ p - t
        if np.max(np.abs(grad)) < 1e-14:
            break
        hess = (F_ * p) @ F_.T - np.outer(F_ @ p, F_ @ p)
        step = -np.linalg.solve(hess, grad)
        f0, a = dual(lam), 1.0
        while dual(lam + a * step) > f0 + 1e-4 * a * (grad @ step) and a > 1e-10:
            a *= 0.5
        lam = lam + a * step
    z = F_.T @ lam
    p = np.exp(z - z.max()); p /= p.sum()
    w = np.concatenate([[w0, w1], mass * p])
    ages = np.arange(d.length)
    check = np.array([w.sum(), ((-1.0) ** ages) @ w, ages @ w, np.log1p(ages) @ w])
    assert np.allclose(check, [1.0, 0.0, d.mean_lag, d.log_moment], rtol=0.0, atol=1e-9), check
    assert w.min() > 0.0 and len(w) == d.length, (w.min(), len(w))
    return torch.as_tensor(w, dtype=torch.float32)

# I tuned the desiderata for an early momentum kernel and the ending momentum kernel. We solve for each.
# For checkpoints in the middle, we linearly interpolate.
# (The solve reproduces the kernels the n=8 runs used, stored as literals in the submitted logs' script, to 5e-12.)
EARLY_MOMENTUM_DESIDERATA = MomentumDesiderata(mean_lag=90.0, log_moment=3.7153849427, newest_weight=0.07235, second_newest_weight=0.02235)
LATE_MOMENTUM_DESIDERATA = MomentumDesiderata(mean_lag=24.05, log_moment=2.3786333904, newest_weight=0.137, second_newest_weight=0.087)
early_momentum_kernel = solve_for_momentum_kernel_given_desiderata(EARLY_MOMENTUM_DESIDERATA)
late_momentum_kernel = solve_for_momentum_kernel_given_desiderata(LATE_MOMENTUM_DESIDERATA)

def momentum_anneal_fraction(step: int) -> float:
    """Position of `step` in the anneal: 0 at MAXENTSLOP_START (early kernel), 1 at step TRAIN_STEPS. The last executed
    update is step TRAIN_STEPS - 1, so the late kernel is approached over the fixed clock and not quite reached."""
    return min(1.0, max(0.0, (step - MAXENTSLOP_START) / MAXENTSLOP_ANNEAL_STEPS))

def interpolate_momentum_kernels(early_momentum_kernel: MomentumKernel, late_momentum_kernel: MomentumKernel,
                                 fraction: float) -> MomentumKernel:
    """The kernel in force at a step: the entrywise linear interpolation (1 - fraction) * early + fraction * late."""
    return early_momentum_kernel.lerp(late_momentum_kernel, fraction)

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

class RawGradientHistory:
    """The last `length` raw gradients of one parameter, oldest overwritten first.

    `buffer` is a (length, numel) float32 tensor, one flattened gradient per row. Rows are written cyclically:
    the gradient of step s lives in row s % length, so `newest_row` holds the age-0 gradient, the row before it
    the age-1 gradient, and so on, wrapping around. Rows not yet written (the first `length` steps) are zero.
    """
    def __init__(self, length: int, numel: int, device: torch.device):
        self.length = length
        self.buffer = torch.zeros(length, numel, dtype=torch.float32, device=device)
        self.newest_row = length - 1          # so the first push lands in row 0

    def push(self, grad: Tensor) -> None:
        """Record this step's gradient as the age-0 entry."""
        self.newest_row = (self.newest_row + 1) % self.length
        self.buffer[self.newest_row].copy_(grad.flatten().float())

@torch.compile
def _weighted_sum_over_history(momentum_kernel: MomentumKernel, buffer: Tensor, newest_row: int) -> Tensor:
    length = buffer.size(0)
    row_ages = (newest_row - torch.arange(length, device=buffer.device)) % length   # age of the gradient in each row
    return momentum_kernel[row_ages] @ buffer

def apply_momentum_kernel_to_raw_gradient_history(momentum_kernel: MomentumKernel,
                                                  raw_gradient_history: RawGradientHistory) -> Tensor:
    """The momentum direction: sum over ages k of momentum_kernel[k] * (the gradient k steps old). Shape (numel,)."""
    return _weighted_sum_over_history(momentum_kernel, raw_gradient_history.buffer, raw_gradient_history.newest_row)

@torch.compile
def muon_update_maxentslop(grad: Tensor, momentum_direction: Tensor) -> Tensor:
    """MuonH's update from a momentum direction: Newton-Schulz orthogonalisation and the aspect-ratio scale, as in muon_update."""
    update = zeropower_via_newtonschulz5(momentum_direction.view_as(grad))
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
        self.early_momentum_kernel = early_momentum_kernel
        self.late_momentum_kernel = late_momentum_kernel
        self.history_length = MAXENTSLOP_HISTORY_LENGTH

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
                        state["history"] = RawGradientHistory(self.history_length, p.numel(), p.device)
                    state["history"].push(p.grad)
                    if self._step >= MAXENTSLOP_START:
                        if "early_kernel_gpu" not in state:
                            state["early_kernel_gpu"] = self.early_momentum_kernel.to(device=p.device)
                            state["late_kernel_gpu"] = self.late_momentum_kernel.to(device=p.device)
                        momentum_kernel = interpolate_momentum_kernels(
                            state["early_kernel_gpu"], state["late_kernel_gpu"], momentum_anneal_fraction(self._step))
                        momentum_direction = apply_momentum_kernel_to_raw_gradient_history(momentum_kernel, state["history"])
                        momentum_sum = momentum_direction.view_as(p).float()
                        parameter = p.float()
                        parameter_norm_squared = parameter.square().sum()
                        radial_coefficient = (momentum_sum * parameter).sum() / parameter_norm_squared
                        tangent = momentum_sum - radial_coefficient * parameter
                        target_coefficient = (RADIAL_TO_TANGENT_RATIO * torch.sign(radial_coefficient)
                                              * tangent.norm() / parameter_norm_squared.sqrt())
                        momentum_direction = (tangent + target_coefficient * parameter).flatten()
                        update = muon_update_maxentslop(p.grad, momentum_direction)
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
    logfile = (f"logs/muonh_fsd_maxentslop_warm{WARMUP_END}{_plateau_tag}_peak{PEAK_LR}"
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
print0(f"MaxEntSlop: history_length={MAXENTSLOP_HISTORY_LENGTH}, start={MAXENTSLOP_START}, "
       f"early kernel {EARLY_MOMENTUM_DESIDERATA}, late kernel {LATE_MOMENTUM_DESIDERATA}")
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
