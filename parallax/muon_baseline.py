import os as _os
_lr=_os.environ.get('LOCAL_RANK','0')
_os.environ['TMPDIR']='/var/tmp'
_os.environ['TORCHINDUCTOR_CACHE_DIR']=f'/var/tmp/vp_ind_{_lr}'
_os.environ['TRITON_CACHE_DIR']=f'/var/tmp/vp_tri_{_lr}'
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
_sys.path.insert(0, _os.environ.get('PARALLAX_PATH', ''))
from parallax.triton.parallax_func import parallax_func
@torch.compiler.disable
def _parallax_attn(q,r,k,v,s):
    return parallax_func(q,r,k,v,s)
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
        y = _parallax_attn(q.transpose(1,2).bfloat16(), r.transpose(1,2).bfloat16(),
                           k.transpose(1,2).bfloat16(), v.transpose(1,2).bfloat16(), 0.12).transpose(1,2).type_as(x)
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

# @torch.compile  # eager
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
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
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])






def gram_eigenbasis(C: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Eigenbasis for symmetric PSD Gram matrix.
    Returns eigenvectors sorted by descending eigenvalue.
    """
    C = C.float()
    C = 0.5 * (C + C.T)

    if eps > 0:
        eye = torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
        C = C + eps * eye

    evals, evecs = torch.linalg.eigh(C)
    idx = torch.argsort(evals, descending=True)
    return evecs[:, idx]


def sinkhorn_energy_balance_rect_no_scale(
    A: Tensor,
    steps: int = 10,
    eps: float = 1e-8,
) -> Tensor:
    """
    Sinkhorn-balance elementwise energy A^2.

    For A in R^{m x n}:

        B = A^2 + eps
        Pi = Dr B Dc

    target rectangular marginals:

        Pi 1_n ~= (1/m) 1_m
        Pi^T 1_m ~= (1/n) 1_n

    Lift back:

        A_bal = Dr^{1/2} A Dc^{1/2}

    No RMS/Frobenius scaling here.
    """
    assert A.ndim == 2

    out_dtype = A.dtype
    A = A.float()

    m, n = A.shape
    B = A.square() + eps

    target_r = torch.full(
        (m,),
        1.0 / m,
        device=A.device,
        dtype=torch.float32,
    )
    target_c = torch.full(
        (n,),
        1.0 / n,
        device=A.device,
        dtype=torch.float32,
    )

    r = torch.ones(m, device=A.device, dtype=torch.float32)
    c = torch.ones(n, device=A.device, dtype=torch.float32)

    for _ in range(steps):
        r = target_r / (B @ c + eps)
        c = target_c / (B.T @ r + eps)

    A_bal = r.sqrt()[:, None] * A * c.sqrt()[None, :]

    return A_bal.to(out_dtype)


def scale_update_like_muon(update: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Match original Muon's final Frobenius norm:

        ||update||_F ~= sqrt(min(m, n)) * sqrt(max(1, m / n))

    This equals sqrt(m), but we keep the Muon expression.
    """
    assert update.ndim == 2

    m, n = update.shape

    target_norm = (min(m, n) ** 0.5) * (max(1.0, m / n) ** 0.5)

    update_norm = update.float().norm(dim=(-2, -1), keepdim=True).clamp_min(eps)

    return update * (target_norm / update_norm).to(update.dtype)


def normuon_postcondition_update(
    update: Tensor,
    second_momentum: Tensor,
    beta: float = 0.95,
    eps: float = 1e-10,
) -> Tensor:
    """
    NorMuon-style postconditioner applied to an already preconditioned update.

    Important:
    - This does NOT call zeropower again.
    - It preserves the incoming Frobenius norm.
    - Therefore call this AFTER scale_update_like_muon(update).

    For m >= n:
        v_mean shape [m, 1], row-wise energy.

    For m < n:
        v_mean shape [1, n], column-wise energy.
    """
    assert update.ndim == 2

    out_dtype = update.dtype
    X = update.float()

    m, n = X.shape

    norm = X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)

    if m >= n:
        v_mean = torch.mean(X * X, dim=-1, keepdim=True)  # [m, 1]
    else:
        v_mean = torch.mean(X * X, dim=-2, keepdim=True)  # [1, n]

    second_momentum.lerp_(v_mean.to(second_momentum.dtype), 1.0 - beta)

    step_size = 1.0 / second_momentum.float().sqrt().clamp_min(eps)

    X = X * step_size

    norm_new = X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    X = X * (norm / norm_new)

    return X.to(out_dtype)


def gram_sinkhorn_normuon_update(
    grad: Tensor,
    momentum: Tensor,
    left_gram: Tensor,
    right_gram: Tensor,
    second_momentum: Tensor,
    mu: float = 0.95,
    gram_beta: float = 0.95,
    normuon_beta: float = 0.95,
    nesterov: bool = True,
    sinkhorn_steps: int = 10,
    eps: float = 1e-8,
    normuon_eps: float = 1e-10,
    apply_normuon: bool = True,
) -> Tensor:
    """
    Gram-Sinkhorn update + NorMuon postconditioner.

    Order:

        1. momentum
        2. EMA GG^T and G^T G
        3. U, V from Gram eigenspaces
        4. A = U^T M V
        5. Sinkhorn on A^2
        6. update = U A_bal V^T
        7. scale update to Muon Frobenius norm
        8. NorMuon postconditioner, preserving that norm
    """
    assert grad.ndim == 2
    assert momentum.shape == grad.shape

    G = grad.float()
    m, n = G.shape

    # Momentum EMA.
    momentum.lerp_(grad, 1.0 - mu)

    if nesterov:
        M = grad.float().lerp(momentum.float(), mu)
    else:
        M = momentum.float()

    # EMA Gram matrices.
    left_gram.lerp_(G @ G.T, 1.0 - gram_beta)
    right_gram.lerp_(G.T @ G, 1.0 - gram_beta)

    # Bases from EMA Grams.
    U = gram_eigenbasis(left_gram, eps=eps)
    V = gram_eigenbasis(right_gram, eps=eps)

    # Project momentum/update into Gram feature basis.
    A = U.T @ M @ V

    # Sinkhorn energy balancing, no scale restoration.
    A_bal = sinkhorn_energy_balance_rect_no_scale(
        A,
        steps=sinkhorn_steps,
        eps=eps,
    ).float()

    # Reconstruct matrix update.
    update = U @ A_bal @ V.T

    # First align final matrix update scale to Muon.
    update = scale_update_like_muon(update, eps=eps)

    # Then apply NorMuon-style postconditioner.
    # It preserves the norm, so final scale remains Muon-aligned.
    if apply_normuon and (m != n):
        update = normuon_postcondition_update(
            update,
            second_momentum=second_momentum,
            beta=normuon_beta,
            eps=normuon_eps,
        )

    return update.to(grad.dtype)


class SinkSOAP(torch.optim.Optimizer):
    """
    Gram-Sinkhorn-Muon + NorMuon postconditioner.

    State per 2D parameter:

        momentum:        [m, n]
        left_gram:       [m, m]
        right_gram:      [n, n]
        second_momentum: [m, 1] if m >= n else [1, n]

    Final update scale is aligned to original Muon Frobenius norm before
    NorMuon postconditioning.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
        gram_beta: float = 0.95,
        normuon_beta: float = 0.95,
        sinkhorn_steps: int = 10,
        nesterov: bool = True,
        eps: float = 1e-8,
        normuon_eps: float = 1e-10,
        apply_normuon: bool =True,
    ):
        assert isinstance(params, list)
        assert len(params) >= 1
        assert isinstance(params[0], torch.nn.Parameter)

        params = sorted(params, key=lambda x: x.size(), reverse=True)

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            gram_beta=gram_beta,
            normuon_beta=normuon_beta,
            sinkhorn_steps=sinkhorn_steps,
            nesterov=nesterov,
            eps=eps,
            normuon_eps=normuon_eps,
            apply_normuon=apply_normuon,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        for group in self.param_groups:
            params = group["params"]

            if world_size > 1:
                pad_len = (world_size - len(params) % world_size) % world_size
                params_pad = params + [torch.empty_like(params[-1])] * pad_len
            else:
                params_pad = params

            for base_i in range(0, len(params_pad), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]

                    if p.grad is None:
                        continue

                    if p.grad.ndim != 2:
                        raise ValueError(
                            "GramSinkhornNorMuon currently only supports 2D parameters. "
                            f"Got parameter with shape {tuple(p.shape)}."
                        )

                    state = self.state[p]
                    m, n = p.shape

                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)

                        state["left_gram"] = torch.zeros(
                            (m, m),
                            device=p.device,
                            dtype=torch.float32,
                        )

                        state["right_gram"] = torch.zeros(
                            (n, n),
                            device=p.device,
                            dtype=torch.float32,
                        )

                        if m >= n:
                            state["second_momentum"] = torch.zeros(
                                (m, 1),
                                device=p.device,
                                dtype=torch.float32,
                            )
                        else:
                            state["second_momentum"] = torch.zeros(
                                (1, n),
                                device=p.device,
                                dtype=torch.float32,
                            )

                    update = gram_sinkhorn_normuon_update(
                        grad=p.grad,
                        momentum=state["momentum"],
                        left_gram=state["left_gram"],
                        right_gram=state["right_gram"],
                        second_momentum=state["second_momentum"],
                        mu=group["mu"],
                        gram_beta=group["gram_beta"],
                        normuon_beta=group["normuon_beta"],
                        nesterov=group["nesterov"],
                        sinkhorn_steps=group["sinkhorn_steps"],
                        eps=group["eps"],
                        normuon_eps=group["normuon_eps"],
                        apply_normuon=group["apply_normuon"],
                    )

                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )


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
mbs = 16  # eager memory; grad-accum keeps effective batch
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
# model.compile(dynamic=False)  # eager


num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for _ in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = 3500  # generous: vanilla Muon crosses ~3300-3400

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=0.04, weight_decay=0.025, mu=0.95)
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
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
        val_step_freq = 125 if step / train_steps < 0.94 else 25
        if step == train_steps or step % val_step_freq == 0 or step == 3080 or step == 3085 or step == 3090:
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
