"""
init_program.py — Modded-NanoGPT Optimization Benchmark (Track 3).

Derived from records/track_3_optimization/train_gpt_simple.py at the
upstream repo (https://github.com/KellerJordan/modded-nanogpt).

Goal: minimize the first optimization step that reaches validation loss
< 3.278 on the FineWeb val set. Candidate code does not control the
stopping point; the benchmark loop keeps training until the target is
reached, then stops and reports that first target step.

Rules (track 3):
1. Do NOT modify the dataset, batch size, or architecture.
2. The trainer must not perform more than one forward-backward pass per step.
3. Validation loss must satisfy val_loss < 3.278.
4. No candidate-controlled early stopping based on val loss or step count.

Free to modify (inside the EVOLVE-BLOCK below):
- The optimizer algorithm (e.g. Muon variants, AdamW variants, SOAP, ...).
- All optimization hyperparameters (lr, wd, betas, schedules, init std, ...).
- The model parameter initialization scheme.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path(filename_pattern).parent.glob(Path(filename_pattern).name))
    assert files, f"no files matched pattern {filename_pattern}"
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
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# data path; default mirrors the upstream repo layout, overridable for hosted eval
DATA_ROOT = os.environ.get("MODDED_NANOGPT_DATA", "data/fineweb10B")

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

# we begin by logging this file itself
print0(code)
print0("=" * 100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("=" * 100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator(f"{DATA_ROOT}/fineweb_val_*.bin", val_tokens))

SEED = int(os.environ.get("MODDED_NANOGPT_SEED", "0"))
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)

num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

# Benchmark-owned stopping policy. Candidate code may tune schedules, but it
# must not decide when training ends.
TARGET_VAL_LOSS = 3.278
BENCHMARK_STEP_CAP = int(os.environ.get("MODDED_NANOGPT_STEP_CAP", "5000"))
FINE_VAL_START_STEP = int(os.environ.get("MODDED_NANOGPT_FINE_VAL_START_STEP", "2500"))
FINE_VAL_STEP_FREQ = int(os.environ.get("MODDED_NANOGPT_FINE_VAL_STEP_FREQ", "10"))
STOP_AT_TARGET = os.environ.get("MODDED_NANOGPT_STOP_AT_TARGET", "0") == "1"
SAVE_CHECKPOINTS = os.environ.get("MODDED_NANOGPT_SAVE_CHECKPOINTS", "0") == "1"
CHECKPOINT_DIR = Path(os.environ.get(
    "MODDED_NANOGPT_CHECKPOINT_DIR",
    "records/track_3_optimization/results/20260622_tailema120_n4_prelim/checkpoints",
))
train_steps = BENCHMARK_STEP_CAP


# EVOLVE-BLOCK-START
# ============================================================================
# Optimization + Init & Optim Hyperparams
#
# This is the only region of the program that may change between candidates.
# Below it, the training loop reads `optimizers`, `set_hparams(step)`,
# and the freshly initialized `model` parameters.
#
# Required outputs from this block (by name, evaluated once per trial):
#   - optimizers  : list[torch.optim.Optimizer]
#                                  Must cover exactly model.parameters().
#                                  Each param group must define "initial_lr"
#                                  (set to its starting lr).
#   - set_hparams : Callable[[int], None]
#                                  Called once per step before opt.step();
#                                  responsible for updating learning rates
#                                  (and any other per-step hparams).
# It must also have just (re-)initialized `model.named_parameters()`.
#
# The benchmark owns `SEED`; candidate code may use it for deterministic
# initialization, but must not rely on a hard-coded single seed.
#
# Do not define a stopping step in this block. The benchmark owns `train_steps`
# and stops only after val_loss < TARGET_VAL_LOSS, with BENCHMARK_STEP_CAP as a
# safety cap for broken/divergent candidates.
# ============================================================================

import torch.distributed as dist
from torch import Tensor

FINAL_SCHEDULE_STEPS = 2900
MUON_SCHEDULE_STEPS = 2875
FINAL_TRAIN_STEPS = 2900
FINAL_LR_POWER = 1.2
MUON_LR_POWER = 1.183
ADAM_EMBED_POWER_C = 4.976805410800738e-05
ADAM_PROJ_POWER_C = 5.184172302917436e-07
ADAM_OTHER_POWER_C = 1.6589351369335795e-06
MUON_POWER_C = 3.3169534699576625e-06
MU = 0.95
MUON_LR = 0.0375
TARGET_UW = 0.3825
SOAP_TARGET_UW = TARGET_UW
NONSOAP_TARGET_UW = TARGET_UW
SOAP_BETA2 = 0.90
SOAP_PRECONDITION_FREQUENCY = 1
SOAP_DENOM_POWER = 0.50
ATTN_EARLY_TRUST_FLOOR = 0.45
ATTN_EARLY_TRUST_CAP = 0.85
ATTN_TRUST_FLOOR_END_STEP = 1375
ATTN_TRUST_FLOOR_FADE_END_STEP = 1625
ATTN_TRUST_MIN_AGREE = 0.20
ATTN_TRUST_MIN_GRAD_ALIGN = 0.00
ATTN_TRUST_POWER = 1.00
RADIAL_OUTWARD_SCALE = 0.5
RADIAL_INWARD_SCALE = 1.0
HEAD_DIM = 128
ROWFLOOR = True
ROWFLOOR_RHO = 1.0
CWD = 0.025
TAILEMA_TAU = 120.0
TAILEMA_START = 2400
TAILEMA_END = 2900
TAILEMA_LAMBDA = 0.65

def gram_frobenius_norm_estimate(G: Tensor, keepdim: bool = False, eps: float = 1e-10) -> Tensor:
    X = G.float()
    gram = X.mT @ X if X.size(-2) > X.size(-1) else X @ X.mT
    return gram.norm(dim=(-2, -1), keepdim=keepdim).sqrt().clamp_min(eps)

def _ns_inner(X: Tensor) -> Tensor:
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / gram_frobenius_norm_estimate(X, keepdim=True, eps=1e-7).to(X.dtype)
    X = _ns_inner(X)
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def should_soap_param(name: str) -> bool:
    return (
        name.endswith(".mlp.fc.weight")
        or name.endswith(".mlp.proj.weight")
        or name.endswith(".attn.q.weight")
        or name.endswith(".attn.k.weight")
        or name.endswith(".attn.v.weight")
        or name.endswith(".attn.proj.weight")
    )

def is_attn_proj_param(name: str) -> bool:
    return name.endswith(".attn.proj.weight")

def is_attn_param(name: str) -> bool:
    return (
        name.endswith(".attn.q.weight")
        or name.endswith(".attn.k.weight")
        or name.endswith(".attn.v.weight")
        or name.endswith(".attn.proj.weight")
    )

def tensor_cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a_f, b_f = a.float(), b.float()
    return (a_f * b_f).sum() / (a_f.norm() * b_f.norm()).clamp_min(eps)

def trust_gate(raw: Tensor, soap: Tensor, grad: Tensor, eps: float = 1e-8) -> Tensor:
    raw_grad = tensor_cosine(raw, grad, eps)
    soap_grad = tensor_cosine(soap, grad, eps)
    soap_raw = tensor_cosine(soap, raw, eps)
    agree_gate = ((soap_raw - ATTN_TRUST_MIN_AGREE) / (1 - ATTN_TRUST_MIN_AGREE)).clamp(0, 1)
    denom = (raw_grad - ATTN_TRUST_MIN_GRAD_ALIGN).clamp_min(eps)
    grad_gate = ((soap_grad - ATTN_TRUST_MIN_GRAD_ALIGN) / denom).clamp(0, 1)
    gate = (agree_gate * grad_gate).clamp(0, 1)
    return gate.pow(ATTN_TRUST_POWER) if ATTN_TRUST_POWER != 1.0 else gate

def early_trust_floor_for_step(step: int) -> float:
    if step < ATTN_TRUST_FLOOR_END_STEP:
        return ATTN_EARLY_TRUST_FLOOR
    if step >= ATTN_TRUST_FLOOR_FADE_END_STEP:
        return 0.0
    return ATTN_EARLY_TRUST_FLOOR * (
        ATTN_TRUST_FLOOR_FADE_END_STEP - step
    ) / (ATTN_TRUST_FLOOR_FADE_END_STEP - ATTN_TRUST_FLOOR_END_STEP)

def bounded_trust_gate(gate: Tensor, step: int) -> Tensor:
    floor = early_trust_floor_for_step(step)
    cap = ATTN_EARLY_TRUST_CAP if step < ATTN_TRUST_FLOOR_FADE_END_STEP else 1.0
    return gate.clamp(min=floor, max=cap)

def norm_preserving_blend(raw: Tensor, soap: Tensor, gate: Tensor, eps: float = 1e-8) -> Tensor:
    blended = raw + (soap - raw) * gate.to(raw.dtype)
    raw_norm = gram_frobenius_norm_estimate(raw, eps=eps)
    blended_norm = gram_frobenius_norm_estimate(blended, eps=eps)
    return (blended * (raw_norm / blended_norm).to(blended.dtype)).to(raw.dtype)

def scale_radial_update(update: Tensor, param: Tensor, eps: float = 1e-12) -> Tensor:
    update_f = update.float()
    param_f = param.float()
    denom = (param_f * param_f).sum().clamp_min(eps)
    coeff = (update_f * param_f).sum() / denom
    radial = coeff * param_f
    tangential = update_f - radial
    radial_scale = torch.where(
        coeff < 0,
        update_f.new_tensor(RADIAL_OUTWARD_SCALE),
        update_f.new_tensor(RADIAL_INWARD_SCALE),
    )
    return (tangential + radial_scale * radial).to(update.dtype)

def target_radius_after_update(param: Tensor, update: Tensor, lr: float, eps: float = 1e-8) -> Tensor:
    param_f = param.float()
    update_f = update.float()
    before_norm = param_f.norm().clamp_min(eps)
    radial_delta = -lr * (update_f * param_f).sum() / before_norm
    return (before_norm + radial_delta).clamp_min(eps)

def rescale_to_radius(param: Tensor, target_norm: Tensor, eps: float = 1e-8):
    after_norm = param.float().norm().clamp_min(eps)
    param.mul_((target_norm / after_norm).to(param.dtype))

def soap_eigenbasis(mat: Tensor) -> Tensor:
    try:
        _, q = torch.linalg.eigh(mat + 1e-30 * torch.eye(mat.size(0), device=mat.device))
    except RuntimeError:
        _, q = torch.linalg.eigh(mat.double() + 1e-30 * torch.eye(mat.size(0), device=mat.device))
        q = q.float()
    return torch.flip(q, [1])

def soap_basis_qr(row_gg, col_gg, q_row, q_col, exp_avg_sq):
    row_eig = torch.diag(q_row.T @ row_gg @ q_row)
    row_sort = torch.argsort(row_eig, descending=True)
    q_row = q_row[:, row_sort]
    exp_avg_sq = exp_avg_sq.index_select(0, row_sort)
    q_row, _ = torch.linalg.qr(row_gg @ q_row)
    col_eig = torch.diag(q_col.T @ col_gg @ q_col)
    col_sort = torch.argsort(col_eig, descending=True)
    q_col = q_col[:, col_sort]
    exp_avg_sq = exp_avg_sq.index_select(1, col_sort)
    q_col, _ = torch.linalg.qr(col_gg @ q_col)
    return q_row, q_col, exp_avg_sq

def soap_precondition_momentum(update, state, beta2=SOAP_BETA2, eps=1e-8):
    update_f = update.float()
    if state["q_row"] is None:
        return update
    q_row, q_col = state["q_row"], state["q_col"]
    projected = q_row.T @ update_f @ q_col
    state["exp_avg_sq"].mul_(beta2).add_(projected.square(), alpha=1 - beta2)
    denom = state["exp_avg_sq"].clamp_min(eps * eps).pow(SOAP_DENOM_POWER)
    precond = q_row @ (projected / denom) @ q_col.T
    precond.mul_(gram_frobenius_norm_estimate(update_f, eps=eps) / gram_frobenius_norm_estimate(precond, eps=eps))
    return precond.to(update.dtype)

def soap_update_preconditioner(grad, state, shampoo_beta=SOAP_BETA2, precondition_frequency=SOAP_PRECONDITION_FREQUENCY):
    grad_f = grad.float()
    state["row_gg"].lerp_(grad_f @ grad_f.T, 1 - shampoo_beta)
    state["col_gg"].lerp_(grad_f.T @ grad_f, 1 - shampoo_beta)
    if state["q_row"] is None:
        state["q_row"] = soap_eigenbasis(state["row_gg"])
        state["q_col"] = soap_eigenbasis(state["col_gg"])
    elif state["soap_step"] > 0 and state["soap_step"] % precondition_frequency == 0:
        state["q_row"], state["q_col"], state["exp_avg_sq"] = soap_basis_qr(
            state["row_gg"], state["col_gg"], state["q_row"], state["q_col"], state["exp_avg_sq"]
        )
    state["soap_step"] += 1

def muon_update(update):
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, named_params, lr=0.02, mu=0.95):
        assert isinstance(named_params, list) and len(named_params) >= 1
        self.soap_params = {p for n, p in named_params if should_soap_param(n)}
        self.attn_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_param(n)}
        self.attn_proj_soap_params = {p for n, p in named_params if should_soap_param(n) and is_attn_proj_param(n)}
        self.step_count = 0
        params = sorted([p for _, p in named_params], key=lambda x: x.size(), reverse=True)
        super().__init__(params, dict(lr=lr, mu=mu))

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
                        if p in self.soap_params:
                            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                            state["row_gg"] = torch.zeros(p.size(0), p.size(0), dtype=torch.float32, device=p.device)
                            state["col_gg"] = torch.zeros(p.size(1), p.size(1), dtype=torch.float32, device=p.device)
                            state["q_row"] = None
                            state["q_col"] = None
                            state["soap_step"] = 0
                    grad = p.grad
                    state["momentum"].lerp_(grad, 1 - group["mu"])
                    momentum_update = grad.lerp(state["momentum"], group["mu"])
                    use_soap = p in self.soap_params
                    if use_soap:
                        if p in self.attn_soap_params:
                            soap_update = soap_precondition_momentum(momentum_update, state)
                            gate = bounded_trust_gate(
                                trust_gate(momentum_update, soap_update, grad),
                                self.step_count,
                            ) if p in self.attn_proj_soap_params else torch.ones((), dtype=torch.float32, device=p.device)
                            momentum_update = norm_preserving_blend(momentum_update, soap_update, gate)
                        else:
                            momentum_update = soap_precondition_momentum(momentum_update, state)
                    update = scale_radial_update(muon_update(momentum_update), p)
                    target_uw = SOAP_TARGET_UW if use_soap else NONSOAP_TARGET_UW
                    if ROWFLOOR and p.ndim == 2:
                        r_row = p.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
                        s_row = update.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
                        f_row = torch.clamp(target_uw * r_row / s_row, min=1.0).pow(ROWFLOOR_RHO)
                        update = (update.float() * f_row).to(update.dtype)
                    else:
                        p_fro = p.float().norm().clamp_min(1e-8)
                        u_fro = update.float().norm().clamp_min(1e-8)
                        scale = torch.where(u_fro / p_fro < target_uw, target_uw * p_fro / u_fro, torch.ones_like(p_fro))
                        update = update * scale.to(update.dtype)
                    target_radius = target_radius_after_update(p, update, group["lr"])
                    if CWD > 0.0 and p.ndim == 2:
                        cwd_mask = (update.float() * p.float() > 0).to(p.dtype)
                    p.add_(update, alpha=-group["lr"])
                    rescale_to_radius(p, target_radius)
                    if CWD > 0.0 and p.ndim == 2:
                        p.mul_(1.0 - (group["lr"] * CWD) * cwd_mask)
                    if use_soap:
                        soap_update_preconditioner(grad, state)
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self.step_count += 1

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.8, 0.99), eps=1e-10):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.lerp_(p.grad, 1 - beta1)
                exp_avg_sq.lerp_(p.grad.float().square(), 1 - beta2)
                adam_dir = exp_avg.float() / (exp_avg_sq.sqrt() + eps)
                p.add_(adam_dir.to(p.dtype), alpha=-group["lr"])

class EMA_Nesterov(torch.optim.Optimizer):
    def __init__(self, params, inner_optimizer, lookahead_stepsize=0, use_scheduled_lookahead_stepsize=True,
                 lookahead_ema=0.9, prefill_steps=0, rest_steps=0):
        super().__init__(params, {})
        self.inner_optimizer = inner_optimizer
        self.use_scheduled_lookahead_stepsize = use_scheduled_lookahead_stepsize
        self.lookahead_stepsize = lookahead_stepsize
        self.lookahead_ema = lookahead_ema
        self.prefill_steps = prefill_steps
        self.rest_steps = rest_steps
        self.it = 0
        self.lookahead_status = False
        self.current_lookahead_stepsize = 0
        self.initialize_buffers()

    @torch.no_grad()
    def initialize_buffers(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "prev_params" not in state:
                    state["prev_params"] = (p.clone(), self.it)
                if "lookahead_buffer" not in state:
                    state["lookahead_buffer"] = (torch.zeros_like(p), -1)

    def get_lr_lambda(self):
        return self.inner_optimizer[0].param_groups[0]["lr"] / self.inner_optimizer[0].param_groups[0]["initial_lr"]

    @torch.no_grad()
    def nesterov_step(self):
        if self.it + 1 > self.prefill_steps and self.it < self.rest_steps and not self.lookahead_status:
            lookahead_stepsize = self.lookahead_stepsize * self.get_lr_lambda() if self.use_scheduled_lookahead_stepsize else self.lookahead_stepsize
            self.current_lookahead_stepsize = lookahead_stepsize
            for group in self.param_groups:
                for p in group["params"]:
                    p.add_(self.state[p]["lookahead_buffer"][0], alpha=lookahead_stepsize)
        else:
            self.current_lookahead_stepsize = 0
        self.lookahead_status = True

    @torch.no_grad()
    def accum_lookahead(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                look = p.add(state["prev_params"][0], alpha=-1)
                buf = state["lookahead_buffer"][0]
                state["lookahead_buffer"] = (buf.lerp_(look, 1 - self.lookahead_ema), self.it)
                state["prev_params"] = (state["prev_params"][0].copy_(p), self.it)

    @torch.no_grad()
    def step(self):
        if not self.lookahead_status:
            self.nesterov_step()
        for opt in self.inner_optimizer:
            opt.step()
        self.accum_lookahead()
        _maybe_update_tailema(self.it + 1)
        self.lookahead_status = False
        self.it += 1

for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()

_DI_FC_ALPHA = 0.30
_NUM_BLOCKS = len(model.blocks)
with torch.no_grad():
    for l_idx, block in enumerate(model.blocks):
        ramp = l_idx / (_NUM_BLOCKS - 1) if _NUM_BLOCKS > 1 else 0.0
        block.mlp.fc.weight.data.mul_(1.0 - _DI_FC_ALPHA * ramp)

_CGI_ALPHA = 0.125
_CGI_PAIR_FROM_LAYER = 8
_CGI_HEAD_MEAN_SHRINK = 0.00

def headmean_antithetic_pair(shape, *, device, layer_idx: int, seed: int, head_dim: int = HEAD_DIM):
    width = shape[-1]
    if width % head_dim != 0:
        s0 = torch.randint(0, 2, shape, device=device, dtype=torch.float32) * 2 - 1
        return s0, -s0
    gen = torch.Generator(device=device)
    gen.manual_seed(0xA9170000 + seed * 1009 + layer_idx * 9176)
    heads0, heads1 = [], []
    for _ in range(width // head_dim):
        h0 = (torch.randint(0, 2, (head_dim,), device=device, generator=gen, dtype=torch.int64) * 2 - 1).float()
        plus0 = int((h0 > 0).sum().item())
        plus0 = int(round(head_dim / 2 + _CGI_HEAD_MEAN_SHRINK * (plus0 - head_dim / 2)))
        plus0 = max(0, min(head_dim, plus0))
        h0 = torch.cat([torch.ones(plus0, device=device), -torch.ones(head_dim - plus0, device=device)])
        h0 = h0[torch.randperm(head_dim, device=device, generator=gen)]
        plus1 = head_dim - plus0
        h1 = torch.cat([torch.ones(plus1, device=device), -torch.ones(head_dim - plus1, device=device)])
        h1 = h1[torch.randperm(head_dim, device=device, generator=gen)]
        heads0.append(h0)
        heads1.append(h1)
    return torch.cat(heads0).reshape(shape), torch.cat(heads1).reshape(shape)

with torch.no_grad():
    pair_next = None
    for l_idx, block in enumerate(model.blocks):
        if l_idx >= _CGI_PAIR_FROM_LAYER and (l_idx - _CGI_PAIR_FROM_LAYER) % 2 == 1:
            s = pair_next
            pair_next = None
        elif l_idx >= _CGI_PAIR_FROM_LAYER:
            s, pair_next = headmean_antithetic_pair(
                block.norm1.gains.shape,
                device=block.norm1.gains.device,
                layer_idx=l_idx,
                seed=SEED,
            )
        else:
            s = torch.randint(0, 2, block.norm1.gains.shape, device=block.norm1.gains.device, dtype=torch.float32) * 2 - 1
        block.norm1.gains.data.copy_((1.0 - _CGI_ALPHA * s).to(block.norm1.gains.dtype))
        block.norm2.gains.data.copy_((1.0 + _CGI_ALPHA * s).to(block.norm2.gains.dtype))

optimizer1 = AdamW([
    dict(params=[model.embed.weight], lr=0.3),
    dict(params=[model.proj.weight], lr=1 / 320),
], betas=(0.8, 0.99), eps=1e-10, weight_decay=0, fused=True)
gain_aux_params = [p for n, p in model.named_parameters() if p.ndim < 2 and n.endswith(".gains")]
attn_proj_bias_params = [p for n, p in model.named_parameters() if n.endswith(".attn.proj.bias")]
other_aux_params = [
    p for n, p in model.named_parameters()
    if p.ndim < 2 and not n.endswith(".gains") and not n.endswith(".attn.proj.bias")
]
optimizer3 = Adam([
    dict(params=gain_aux_params, lr=0.01, betas=(0.8, 0.99)),
    dict(params=other_aux_params, lr=0.01, betas=(0.8, 0.997)),
    dict(params=attn_proj_bias_params, lr=0.01, betas=(0.8, 0.9965)),
], lr=0.01, betas=(0.8, 0.99), eps=1e-10)
optimizer2 = Muon([(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2], lr=MUON_LR, mu=MU)
inner_optimizers = [optimizer1, optimizer2, optimizer3]
for opt in inner_optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
optimizer1.param_groups[0]["power_c"] = ADAM_EMBED_POWER_C
optimizer1.param_groups[1]["power_c"] = ADAM_PROJ_POWER_C
optimizer3.param_groups[0]["power_c"] = ADAM_OTHER_POWER_C
optimizer3.param_groups[1]["power_c"] = ADAM_OTHER_POWER_C
optimizer3.param_groups[2]["power_c"] = ADAM_OTHER_POWER_C
optimizer2.param_groups[0]["power_c"] = MUON_POWER_C
optimizer2.param_groups[0]["schedule_steps"] = MUON_SCHEDULE_STEPS
optimizer2.param_groups[0]["schedule_power"] = MUON_LR_POWER

optimizers = [EMA_Nesterov(
    [p for p in model.parameters()],
    inner_optimizers,
    lookahead_stepsize=0.325,
    use_scheduled_lookahead_stepsize=True,
    lookahead_ema=0.99,
    prefill_steps=300,
    rest_steps=FINAL_TRAIN_STEPS - 950,
)]
for group in optimizers[0].param_groups:
    group["initial_lr"] = 1.0

_inner_param_set = set(p for opt in inner_optimizers for group in opt.param_groups for p in group["params"])
assert _inner_param_set == set(model.parameters())
assert set(p for opt in optimizers for group in opt.param_groups for p in group["params"]) == set(model.parameters())

_tailema_params = list(model.parameters())
_tailema = None
_tailema_blended = False
_tailema_stash = None

@torch.no_grad()
def _maybe_update_tailema(done_step):
    global _tailema
    if TAILEMA_TAU <= 0 or done_step < TAILEMA_START or done_step >= TAILEMA_END:
        return
    if _tailema is None:
        _tailema = [
            None if p is model.embed.weight else p.detach().float().clone()
            for p in _tailema_params
        ]
    else:
        for j, p in enumerate(_tailema_params):
            if _tailema[j] is not None:
                _tailema[j].add_(p.detach().float() - _tailema[j], alpha=1.0 / TAILEMA_TAU)

@torch.no_grad()
def _tailema_apply_for_eval():
    global _tailema_blended, _tailema_stash
    if _tailema is None or _tailema_blended:
        return
    _tailema_stash = []
    for j, p in enumerate(_tailema_params):
        ema = _tailema[j]
        if ema is not None:
            _tailema_stash.append((p, p.detach().clone()))
            p.copy_(((1.0 - TAILEMA_LAMBDA) * p.float() + TAILEMA_LAMBDA * ema).to(p.dtype))
    _tailema_blended = True

@torch.no_grad()
def _tailema_restore_after_eval():
    global _tailema_blended, _tailema_stash
    if not _tailema_blended:
        return
    for p, stash in _tailema_stash:
        p.copy_(stash)
    _tailema_stash = None
    _tailema_blended = False

_original_eval = model.eval
_original_train = model.train

def _eval_with_tailema():
    result = _original_eval()
    _tailema_apply_for_eval()
    return result

def _train_with_tailema(mode=True):
    if mode:
        _tailema_restore_after_eval()
    return _original_train(mode)

model.eval = _eval_with_tailema
model.train = _train_with_tailema

_compiled_call_impl = getattr(model, "_compiled_call_impl", None)
if _compiled_call_impl is not None:
    def _call_with_nesterov(*args, **kwargs):
        if model.training and not optimizers[0].lookahead_status:
            optimizers[0].nesterov_step()
        return _compiled_call_impl(*args, **kwargs)
    model._compiled_call_impl = _call_with_nesterov
else:
    _original_forward = model.forward
    def _forward_with_nesterov(*args, **kwargs):
        if model.training and not optimizers[0].lookahead_status:
            optimizers[0].nesterov_step()
        return _original_forward(*args, **kwargs)
    model.forward = _forward_with_nesterov

def _lr(step, initial_lr, power_c, power=1.0):
    return min(initial_lr, power_c * max(0.0, FINAL_SCHEDULE_STEPS - step) ** power)

_MU_MIN = 0.85
_MU_MAX = 0.95
_MU_WARMUP_STEPS = 300
_MU_COOLDOWN_STEPS = 200

def _muon_mu_at_step(step, final_steps):
    cd_start = final_steps - _MU_COOLDOWN_STEPS
    if step < _MU_WARMUP_STEPS:
        return _MU_MIN + (step / max(_MU_WARMUP_STEPS, 1)) * (_MU_MAX - _MU_MIN)
    if step > cd_start:
        return _MU_MAX - ((step - cd_start) / max(_MU_COOLDOWN_STEPS, 1)) * (_MU_MAX - _MU_MIN)
    return _MU_MAX

def set_hparams(step):
    assert step >= 0
    mu = _muon_mu_at_step(step, FINAL_TRAIN_STEPS)
    for opt in inner_optimizers:
        for group in opt.param_groups:
            schedule_steps = group.get("schedule_steps", FINAL_SCHEDULE_STEPS)
            schedule_power = group.get("schedule_power", FINAL_LR_POWER)
            group["lr"] = min(
                group["initial_lr"],
                group["power_c"] * max(0.0, schedule_steps - step) ** schedule_power,
            )
    for group in optimizer2.param_groups:
        group["mu"] = mu

# EVOLVE-BLOCK-END

# Candidate snippets from older runs may still assign benchmark-owned names in
# the evolve block. Reset them here so candidate code cannot stop training early
# or move the validation target.
TARGET_VAL_LOSS = 3.278
BENCHMARK_STEP_CAP = int(os.environ.get("MODDED_NANOGPT_STEP_CAP", "5000"))
FINE_VAL_START_STEP = int(os.environ.get("MODDED_NANOGPT_FINE_VAL_START_STEP", "2500"))
train_steps = BENCHMARK_STEP_CAP


########################################
#        Training and Validation       #
########################################

train_loader = distributed_data_generator(f"{DATA_ROOT}/fineweb_train_*.bin", batch_size)
for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
last_val_step = 0
saved_first_pass_checkpoint = False

def save_rank0_checkpoint(tag: str, step: int, val_loss_float: float):
    if not SAVE_CHECKPOINTS or dist.get_rank() != 0:
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "seed": SEED,
        "step": step,
        "val_loss": val_loss_float,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    path = CHECKPOINT_DIR / f"seed{SEED}_{tag}_step{step}_loss{val_loss_float:.5f}.pt"
    torch.save(payload, path)
    print0(f"checkpoint_saved {path}", console=True)

dist.barrier()
t0 = time.perf_counter()
target_reached = False
for step in range(train_steps + 1):

    # --------------- VALIDATION SECTION -----------------
    val_step_freq = 125 if step < FINE_VAL_START_STEP else FINE_VAL_STEP_FREQ
    if step == train_steps or step % val_step_freq == 0:
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
        val_loss_float = float(val_loss)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*step_avg:.2f}ms", console=True)
        model.train()
        if val_loss_float < TARGET_VAL_LOSS:
            print0(f"target_reached step:{step} val_loss:{val_loss_float:.5f}", console=True)
            if not saved_first_pass_checkpoint:
                save_rank0_checkpoint("first_pass", step, val_loss_float)
                saved_first_pass_checkpoint = True
            target_reached = True
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()
        if target_reached and STOP_AT_TARGET:
            break

    if step == train_steps:
        if target_reached:
            print0(f"fixed_step_complete max_step:{train_steps} val_loss:{val_loss_float:.5f}", console=True)
        else:
            print0(f"target_not_reached max_step:{train_steps} val_loss:{val_loss_float:.5f}", console=True)
        save_rank0_checkpoint("final", step, val_loss_float)
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
