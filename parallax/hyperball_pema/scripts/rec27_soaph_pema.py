import os as _os
_lr=_os.environ.get('LOCAL_RANK','0')
_jid=_os.environ.get('SLURM_JOB_ID','0')
_os.environ['TMPDIR']=f'/tmp/plx_tmp_{_jid}_{_lr}'
_os.makedirs(_os.environ['TMPDIR'], exist_ok=True)
_os.environ['TORCHINDUCTOR_CACHE_DIR']=f'/tmp/r27_ind_{_jid}_{_lr}'
_os.environ['TRITON_CACHE_DIR']=f'/tmp/r27_tri_{_jid}_{_lr}'
"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from parallax_op import parallax_func
def _parallax_attn(q,r,k,v,sc):
    return parallax_func(q,r,k,v,sc)

# ---- Parallax correction-power logging --------------------------------------
# o^PLX = o^SA - Sigma_KV @ rho. During eval we log ||o^PLX|| / ||o^SA|| (per-(batch,
# token,head) output-vector-norm ratio averaged over batch x tokens x heads, per
# attention layer and overall) to quantify how strongly the
# Parallax rho-correction reshapes the underlying softmax-attention output. Opt-in
# via env LOG_PLX_RATIO=1; gated on module.eval() so it costs nothing in training
# and only the eval graph recompiles.
_LOG_PLX_RATIO = bool(int(os.environ.get("LOG_PLX_RATIO", "0")))
RHO_SCALE = float(os.environ.get("RHO_SCALE", "1.0"))  # multiplier on the (rms-normed) Parallax rho -> scales the Sigma_KV @ rho correction
_PLX_NUM: dict = {}
_PLX_DEN: dict = {}
_ATTN_COUNTER = [0]

def _next_attn_id():
    i = _ATTN_COUNTER[0]
    _ATTN_COUNTER[0] += 1
    return i

def _plx_ratio_accumulate(lid, o_plx, o_sa):
    # per-(batch, token, head) ratio: norm over head_dim (last axis), then summed
    # with its element count so the reported value is the ratio averaged over
    # batch x tokens x heads. o_plx / o_sa are (B, H, T, D).
    rn = o_plx.float().norm(dim=-1)
    sn = o_sa.float().norm(dim=-1).clamp_min(1e-30)
    s = (rn / sn).sum()
    c = torch.full((), float(rn.numel()), device=s.device, dtype=s.dtype)
    _PLX_NUM[lid] = _PLX_NUM.get(lid, 0.0) + s
    _PLX_DEN[lid] = _PLX_DEN.get(lid, 0.0) + c

def _plx_ratio_report():
    import torch.distributed as dist
    per = []
    tot_n = tot_d = None
    for lid in sorted(_PLX_NUM):
        n = _PLX_NUM[lid].detach().clone()
        d = _PLX_DEN[lid].detach().clone()
        if dist.is_initialized():
            dist.all_reduce(n)
            dist.all_reduce(d)
        per.append((n / d.clamp_min(1e-30)).item())
        tot_n = n if tot_n is None else tot_n + n
        tot_d = d if tot_d is None else tot_d + d
    _PLX_NUM.clear()
    _PLX_DEN.clear()
    overall = (tot_n / tot_d.clamp_min(1e-30)).item() if tot_n is not None else float("nan")
    return overall, per
import torch.distributed as dist


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
    # MHA; the Parallax W_R cost is offset by the 3.5x (vs 4x) MLP below, matching
    # the baseline-MHA-without-R param count like the other PLX runs (5*dim^2 attn +
    # 7*dim^2 MLP = 12*dim^2/block, same as the old GQA 4*dim^2 + 4x MLP 8*dim^2).
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.r = Linear(dim, hdim)  # Parallax rho
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)
        self.layer_id = _next_attn_id()

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        r = self.r(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        r = F.rms_norm(r, (r.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        r = self.rotary(r)
        qT = q.transpose(1,2).bfloat16(); kT = k.transpose(1,2).bfloat16()
        vT = v.transpose(1,2).bfloat16(); rT = (r * RHO_SCALE).transpose(1,2).bfloat16()
        o_plx = _parallax_attn(qT, rT, kT, vT, 0.12)
        if _LOG_PLX_RATIO and not self.training:
            o_sa = F.scaled_dot_product_attention(qT, kT, vT, is_causal=True, scale=0.12)
            _plx_ratio_accumulate(self.layer_id, o_plx, o_sa)
        y = o_plx.transpose(1,2).type_as(x)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 7 * dim // 2   # 3.5x param-match (cancels Parallax W_R), like the other PLX runs
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

@torch.no_grad()
def scale_invariant_update_(param: Tensor, update: Tensor, lr: float, eps: float = 1e-10):
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    param.copy_(new_param / torch.clamp(new_param.norm(), min=eps) * p_norm)


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.T)


def _initial_orthogonal_matrix(matrix: Tensor) -> Tensor:
    matrix = _symmetrize(matrix.float())
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=torch.float32)
    try:
        _, q = torch.linalg.eigh(matrix + 1e-30 * eye)
    except RuntimeError:
        _, q = torch.linalg.eigh((matrix + 1e-30 * eye).double())
        q = q.float()
    return torch.flip(q, dims=[1]).contiguous()


@torch.compile
def soap_update(
    grad: Tensor,
    momentum_buffer: Tensor,
    square_buffer: Tensor,
    left_buffer: Tensor,
    right_buffer: Tensor,
    Q_left: Tensor,
    Q_right: Tensor,
    momentum: float,
    ema_beta: float,
    eps: float,
) -> Tensor:
    # Momentum EMA:
    momentum_buffer.lerp_(grad, 1.0 - momentum)

    M = momentum_buffer

    # Matrix second moments:
    left_buffer.mul_(ema_beta).addmm_(
        grad,
        grad.T,
        beta=1.0,
        alpha=1.0 - ema_beta,
    )

    right_buffer.mul_(ema_beta).addmm_(
        grad.T,
        grad,
        beta=1.0,
        alpha=1.0 - ema_beta,
    )

    # Project into current SOAP basis.
    M_eigen = Q_left.T @ M @ Q_right
    G_eigen = Q_left.T @ grad @ Q_right

    # Diagonal second moment in SOAP basis.
    square_buffer.mul_(ema_beta).addcmul_(
        G_eigen,
        G_eigen,
        value=1.0 - ema_beta,
    )

    # Precondition in SOAP basis and rotate back.
    U_eigen = M_eigen / square_buffer.sqrt().add(eps)
    U = Q_left @ U_eigen @ Q_right.T

    return U


class SOAPH(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        ema_beta: float = 0.9,
        momentum: float = 0.95,
        q_freq: int = 1,
        eps: float = 1e-8,
        hyperball_eps: float = 1e-10,
    ):
        assert isinstance(params, list)
        assert len(params) >= 1
        assert isinstance(params[0], torch.nn.Parameter)

        for p in params:
            assert p.ndim == 2, f"SOAPH here expects only 2D parameters, got shape {tuple(p.shape)}"

        params = sorted(params, key=lambda x: x.numel(), reverse=True)

        defaults = dict(
            lr=lr,
            ema_beta=ema_beta,
            momentum=momentum,
            q_freq=q_freq,
            eps=eps,
            hyperball_eps=hyperball_eps,
        )

        print("SOAPH hyperparams:", defaults)

        super().__init__(params, defaults)

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict) -> None:
        m, n = p.shape

        state["step"] = 0

        state["momentum_buffer"] = torch.zeros_like(p)
        state["square_buffer"] = torch.zeros_like(p)

        state["left_buffer"] = torch.zeros(
            (m, m),
            dtype=p.dtype,
            device=p.device,
        )

        state["right_buffer"] = torch.zeros(
            (n, n),
            dtype=p.dtype,
            device=p.device,
        )

        state['Q_left'] = _initial_orthogonal_matrix(p.grad @ p.grad.T)
        state["Q_right"] = _initial_orthogonal_matrix(p.grad.T @ p.grad)

    @torch.no_grad()
    def _refresh_q(self, state: dict) -> None:
        L = state["left_buffer"]
        R = state["right_buffer"]

        Q_left_old = state["Q_left"]
        Q_right_old = state["Q_right"]

        Q_left, _ = torch.linalg.qr(
            L.float() @ Q_left_old.float(),
            mode="reduced",
        )

        Q_right, _ = torch.linalg.qr(
            R.float() @ Q_right_old.float(),
            mode="reduced",
        )

        state["Q_left"].copy_(Q_left.to(dtype=Q_left_old.dtype))
        state["Q_right"].copy_(Q_right.to(dtype=Q_right_old.dtype))

    @torch.no_grad()
    def _step_param(self, p: Tensor, group: dict) -> None:
        if p.grad is None:
            return

        if p.grad.is_sparse:
            raise RuntimeError("SOAPH does not support sparse gradients")

        if p.ndim != 2:
            raise RuntimeError(f"SOAPH expects only 2D parameters, got shape {tuple(p.shape)}")

        grad = p.grad
        state = self.state[p]

        if len(state) == 0:
            self._init_state(p, state)

        lr = group["lr"]
        ema_beta = group["ema_beta"]
        momentum = group["momentum"]
        q_freq = group["q_freq"]
        eps = group["eps"]
        hyperball_eps = group["hyperball_eps"]

        state["step"] += 1
        step = state["step"]

        update = soap_update(
            grad=grad,
            momentum_buffer=state["momentum_buffer"],
            square_buffer=state["square_buffer"],
            left_buffer=state["left_buffer"],
            right_buffer=state["right_buffer"],
            Q_left=state["Q_left"],
            Q_right=state["Q_right"],
            momentum=momentum,
            ema_beta=ema_beta,
            eps=eps,
        )

        if step % q_freq == 0:
            self._refresh_q(state)

        scale_invariant_update_(
            param=p,
            update=update,
            lr=lr,
            eps=hyperball_eps,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()

        if distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        for group in self.param_groups:
            params = group["params"]

            for i, p in enumerate(params):
                owner = i % world_size

                if rank == owner:
                    self._step_param(p, group)

                if distributed:
                    dist.broadcast(p, src=owner)

        return loss



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

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

train_seed = int(os.environ.get("SEED", "1"))
torch.manual_seed(train_seed)
torch.cuda.manual_seed_all(train_seed)
print0(f"Using seed={train_seed}", console=True)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 16  # eager mem; grad-accum neutral
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


num_trials = 1


class ProjectedEMANesterov:
    """Manifold-aware EMA-Nesterov lookahead. flat_params (AdamW: embed/head/scalars) get a
    plain additive lookahead; sphere_params (hyperball 2D matrices) get a lookahead projected to
    the tangent of the Frobenius sphere (radial component removed) then retracted to radius
    r=||W||, so the norm invariant of scale_invariant_update_ is preserved. Accumulates a slow EMA
    of per-iteration displacements and pushes along it before the gradient eval (Nesterov), gated
    to prefill<it<rest and scaled by lr_lambda()."""
    def __init__(self, flat_params, sphere_params, stepsize, ema, prefill_steps, rest_steps, lr_lambda):
        self.flat = list(flat_params)
        self.sphere = list(sphere_params)
        self.a0 = stepsize
        self.beta = ema
        self.prefill = prefill_steps
        self.rest = rest_steps
        self.lr_lambda = lr_lambda
        self.it = 0
        self.cur = 0.0
        self.buf = {p: torch.zeros_like(p) for p in self.flat + self.sphere}
        self.prev = {p: p.detach().clone() for p in self.flat + self.sphere}

    @torch.no_grad()
    def nesterov_step(self):
        a = self.a0 * self.lr_lambda() if (self.prefill < self.it < self.rest) else 0.0
        self.cur = a
        if a == 0.0:
            return
        for p in self.flat:
            p.add_(self.buf[p], alpha=a)
        for p in self.sphere:
            m = self.buf[p]
            r = p.norm().clamp_min(1e-12)
            radial = (p.mul(m).sum() / (r * r)) * p   # <m,W>_F / ||W||_F^2 * W
            cand = p + a * (m - radial)               # step along tangent component only
            p.copy_(cand * (r / cand.norm().clamp_min(1e-12)))  # retract to the r-sphere

    @torch.no_grad()
    def accum(self):
        for p in self.flat + self.sphere:
            self.buf[p].lerp_(p - self.prev[p], 1 - self.beta)
            self.prev[p].copy_(p)
        self.it += 1


for _ in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = int(os.environ.get("TRAIN_STEPS", "3125"))
    lr_steps = int(os.environ.get("LR_STEPS", "3200"))
    lr = 0.018
    lr_power = 1.25
    lr_floor = 0.0


    # initialize model parameters
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    for name, p in model.named_parameters():
        if name.endswith("gains"):
            p.data.fill_(1)
        elif name.endswith(".attn.proj.weight"):
            p.data.mul_(1.25)
        elif name.endswith(".mlp.proj.weight"):
            p.data.mul_(3.0)
        elif name.endswith(".mlp.fc.weight"):
            p.data.mul_(1.5)
        elif name == "proj.weight":
            p.data.zero_()
        elif "proj" in name:
            p.data.zero_()

    # create the optimizer(s)
    named_block_params = [(n, p) for n, p in model.named_parameters()
                          if n.startswith("blocks.") and p.ndim >= 2]
    qkv_params = [p for n, p in named_block_params
                  if n.endswith(".attn.q.weight") or n.endswith(".attn.k.weight") or n.endswith(".attn.v.weight") or n.endswith(".attn.r.weight")]
    mlp_fc_params = [p for n, p in named_block_params if n.endswith(".mlp.fc.weight")]
    attn_proj_params = [p for n, p in named_block_params if n.endswith(".attn.proj.weight")]
    mlp_proj_params = [p for n, p in named_block_params if n.endswith(".mlp.proj.weight")]

    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    optimizer2 = SOAPH(qkv_params, lr=lr)
    optimizer3 = SOAPH(mlp_fc_params, lr=lr)
    optimizer4 = SOAPH(attn_proj_params, lr=lr)
    optimizer5 = SOAPH(mlp_proj_params, lr=lr)
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    for opt in (optimizer2, optimizer3, optimizer4, optimizer5):
        for group in opt.param_groups:
            group["cooldown_frac"] = 1.0
            group["schedule_steps"] = lr_steps
            group["lr_power"] = lr_power
            group["lr_floor"] = lr_floor

    for group in optimizer1.param_groups:
        group["cooldown_frac"] = 0.4
        group["schedule_steps"] = train_steps
        group["lr_power"] = 1.0
        group["lr_floor"] = 0.0


    def _eta(progress: float, cooldown_frac: float, power: float = 1.0, floor: float = 0.0) -> float:
        if progress < 1.0 - cooldown_frac:
            return 1.0

        x = max(0.0, (1.0 - progress) / cooldown_frac)
        return floor + (1.0 - floor) * (x ** power)


    def set_hparams(step):
        for opt in optimizers:
            for group in opt.param_groups:
                schedule_steps = group.get("schedule_steps", train_steps)
                cooldown_frac = group.get("cooldown_frac", 1.0)
                power = group.get("lr_power", 1.0)
                floor = group.get("lr_floor", 0.0)

                progress = step / schedule_steps
                assert 0.0 <= progress < 1.0

                eta = _eta(
                    progress=progress,
                    cooldown_frac=cooldown_frac,
                    power=power,
                    floor=floor,
                )

                group["lr"] = group["initial_lr"] * eta


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    training_time = 0
    last_val_step = 0
    val_interval = 125  # adaptive: -> 25 once val<3.30, -> 5 once val<3.2825 (dense crossing eval)
    dist.barrier()
    t0 = time.perf_counter()
    _ema_lr0 = optimizer2.param_groups[0]["lr"]
    _ema_flat = [p for g in optimizer1.param_groups for p in g["params"]]
    _ema_sphere = [p for opt in optimizers[1:] for g in opt.param_groups for p in g["params"]]
    assert set(_ema_flat) | set(_ema_sphere) == set(model.parameters()), "ema partition mismatch"
    ema = ProjectedEMANesterov(_ema_flat, _ema_sphere, stepsize=float(os.environ.get("EMA_STEPSIZE","0.3")), ema=float(os.environ.get("EMA_DECAY","0.99")),
                               prefill_steps=int(os.environ.get("EMA_PREFILL","323")), rest_steps=train_steps,
                               lr_lambda=lambda: optimizer2.param_groups[0]["lr"] / _ema_lr0)
    for step in range(train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        if step == train_steps or step % val_interval == 0:
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
            if val_loss < 3.2825:
                val_interval = 5     # dense eval to pin the 3.28 crossing
            elif val_loss < 3.30:
                val_interval = 25    # finer as we approach (avoids the 125-grid skipping the crossing)
            if _LOG_PLX_RATIO:
                _plx_overall, _plx_per = _plx_ratio_report()
                print0(f"step:{step}/{train_steps} plx_o_ratio:{_plx_overall:.4f} per_layer:"
                       + ",".join(f"{r:.3f}" for r in _plx_per), console=True)
            print0(f"step:{step}/{train_steps} qnorm:{model.blocks[0].attn.q.weight.norm().item():.5f} ema_a:{ema.cur:.4g}", console=True)
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        set_hparams(step)
        ema.nesterov_step()
        inputs, targets = next(train_loader)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        # set optimization hyperparameters and take a step
        for opt in optimizers:
            opt.step()
        ema.accum()
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

dist.destroy_process_group()
