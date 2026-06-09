import os as _os
_lr=_os.environ.get('LOCAL_RANK','0')
_jid=_os.environ.get('SLURM_JOB_ID','0')
_os.environ['TMPDIR']=f'/tmp/plx_tmp_{_jid}_{_lr}'
_os.makedirs(_os.environ['TMPDIR'], exist_ok=True)
_os.environ['TORCHINDUCTOR_CACHE_DIR']=f'/tmp/rnh_ind_{_jid}_{_lr}'
_os.environ['TRITON_CACHE_DIR']=f'/tmp/rnh_tri_{_jid}_{_lr}'
"""
train_gpt_simple_normuonh.py (Parallax port: rec_normuonh.py)

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.

Differs from `train_gpt_simple.py` only in the init and the optimizer: the matrix-parameter
Muon update is replaced by MuonH (the same Newton-Schulz orthogonalised direction, applied
via a hyperball projection that preserves the Frobenius norm of every hidden 2D weight
matrix at every step), and the residual-side projections / mlp.fc are initialised with
per-module multipliers on the default Kaiming init so MuonH has non-zero matrices to operate
on from step 0.
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
import torch.distributed as dist
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from parallax_op import parallax_func
def _parallax_attn(q,r,k,v,sc):
    return parallax_func(q,r,k,v,sc)

# ---- Parallax correction-power logging --------------------------------------
# o^PLX = o^SA - Sigma_KV @ rho. During eval we log ||o^PLX|| / ||o^SA|| (per-(batch,
# token,head) output-vector-norm ratio averaged over batch x tokens x heads, per
# attention layer and overall). Opt-in via env LOG_PLX_RATIO=1; gated on
# module.eval() so it costs nothing in training and only the eval graph recompiles.
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
    rn = o_plx.float().norm(dim=-1)
    sn = o_sa.float().norm(dim=-1).clamp_min(1e-30)
    s = (rn / sn).sum()
    c = torch.full((), float(rn.numel()), device=s.device, dtype=s.dtype)
    _PLX_NUM[lid] = _PLX_NUM.get(lid, 0.0) + s
    _PLX_DEN[lid] = _PLX_DEN.get(lid, 0.0) + c

def _plx_ratio_report():
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
        qT = q.transpose(1, 2).bfloat16(); kT = k.transpose(1, 2).bfloat16()
        vT = v.transpose(1, 2).bfloat16(); rT = (r * RHO_SCALE).transpose(1, 2).bfloat16()
        o_plx = _parallax_attn(qT, rT, kT, vT, 0.12)
        if _LOG_PLX_RATIO and not self.training:
            o_sa = F.scaled_dot_product_attention(qT, kT, vT, is_causal=True, scale=0.12)
            _plx_ratio_accumulate(self.layer_id, o_plx, o_sa)
        y = o_plx.transpose(1, 2).type_as(x)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 7 * dim // 2   # 3.5x param-match (cancels Parallax W_R)
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
def normuon_update(grad, momentum, v_buf, mu=0.95, beta2=0.95, eps=1e-10, nesterov=True):
    """NorMuon direction: NS-orthogonalised gradient followed by Adafactor-style variance
    preconditioning along the SHORT axis (per https://arxiv.org/pdf/2510.05491). `v_buf`
    is an EMA of squared post-NS values along the short axis, persistent across steps."""
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    if grad.size(-2) >= grad.size(-1):
        v_new = update.square().mean(dim=-1, keepdim=True)
    else:
        v_new = update.square().mean(dim=-2, keepdim=True)
    v_buf.lerp_(v_new.to(v_buf.dtype), 1 - beta2)
    update = update * v_buf.clamp_min(eps).rsqrt()
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

class NorMuonH(torch.optim.Optimizer):
    """NorMuonH: NS-orthogonalised gradient + Adafactor-style row/column variance
    preconditioning (NorMuon) + hyperball Frobenius-norm-preserving step. Used here for
    ALL hidden 2D weight matrices — q, k, v, r, mlp.fc, attn.proj, mlp.proj."""
    def __init__(self, params, lr=0.018, mu=0.95, beta2=0.95, eps=1e-10):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, mu=mu, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

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
                        v_shape = list(p.shape)
                        if p.size(-2) >= p.size(-1):
                            v_shape[-1] = 1
                        else:
                            v_shape[-2] = 1
                        state["v"] = torch.zeros(v_shape, dtype=p.dtype, device=p.device)
                    update = normuon_update(
                        p.grad, state["momentum"], state["v"],
                        mu=group["mu"], beta2=group["beta2"], eps=group["eps"],
                    )
                    scale_invariant_update_(p, update, group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


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

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = int(os.environ.get("MICRO_BATCH_SEQUENCES", "16"))  # parallax mem: grad-accum neutral (loss reduction=sum)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

torch.manual_seed(int(os.environ.get("SEED", "0")))
model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

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
    train_steps = int(os.environ.get("TRAIN_STEPS", "3325"))

    # initialize model parameters. Per-module multipliers on the default nn.Linear Kaiming-uniform
    # init (std = 1/sqrt(3*fan_in), so ~0.0208 for fan_in=768 and ~0.0104 for fan_in=3072):
    #   - attn.proj.weight (fan_in=768):  default × 1.25 → std ≈ 0.026
    #   - mlp.proj.weight  (fan_in=3072): default × 3.0  → std ≈ 0.031
    #   - mlp.fc.weight    (fan_in=768):  default × 1.5  → std ≈ 0.031
    # qkv weights keep their default init. The vocab head (proj.weight) and all "proj" biases are
    # zeroed so initial logits are 0.
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
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    optimizer2 = NorMuonH([p for p in model.blocks.parameters() if p.ndim == 2], lr=0.018)
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    for group in optimizer1.param_groups:
        group["cooldown_frac"] = 0.4
    for group in optimizer2.param_groups:
        group["cooldown_frac"] = 1.0

    # learning rate schedule: stable then decay. The h (MuonH) groups use full linear cooldown
    # over the entire run (cooldown_frac=1.0); the aux (AdamW) group uses a shorter cooldown
    # (cooldown_frac=0.4) to keep the embed/head learning longer before tapering.
    def set_hparams(step):
        progress = step / train_steps
        assert 0 <= progress < 1
        for opt in optimizers:
            for group in opt.param_groups:
                if progress < 1 - group["cooldown_frac"]:
                    eta = 1.0
                else:
                    eta = (1 - progress) / group["cooldown_frac"]
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
