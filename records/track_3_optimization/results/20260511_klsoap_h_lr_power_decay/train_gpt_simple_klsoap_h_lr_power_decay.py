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


def project_to_klsoap_basis(grad: Tensor, state) -> Tensor:
    q_left, q_right = state["Q"]
    return q_left.T @ grad.float() @ q_right


def project_from_klsoap_basis(grad: Tensor, state) -> Tensor:
    q_left, q_right = state["Q"]
    return q_left @ grad.float() @ q_right.T


def init_2d_klsoap_state_(state, grad: Tensor, shampoo_beta: float, init_factor: float):
    grad = grad.detach().float()
    rows, cols = grad.shape
    state["step"] = 0
    state["GG"] = [
        (grad @ grad.T / cols).contiguous(),
        (grad.T @ grad / rows).contiguous(),
    ]
    state["Q"] = [
        _initial_orthogonal_matrix(state["GG"][0]),
        _initial_orthogonal_matrix(state["GG"][1]),
    ]
    inv = init_factor ** -0.5
    state["eigen_sqrt_inv"] = [
        torch.full((rows,), inv, device=grad.device, dtype=torch.float32),
        torch.full((cols,), inv, device=grad.device, dtype=torch.float32),
    ]
    state["exp_avg"] = torch.zeros_like(grad, dtype=torch.float32)
    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)
    state["shampoo_beta"] = shampoo_beta


def _update_eigen_sqrt_inv_(state, diag: Tensor, idx: int, beta: float):
    old_eigen = state["eigen_sqrt_inv"][idx].float().square().reciprocal()
    old_eigen = old_eigen.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    eigen = beta * old_eigen + (1.0 - beta) * diag.detach().float()
    inv_sqrt = eigen.clamp_min(1e-30).rsqrt().clamp(max=4000.0)
    state["eigen_sqrt_inv"][idx] = inv_sqrt.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).contiguous()


@torch.no_grad()
def update_2d_klsoap_preconditioner_(grad: Tensor, state):
    grad = grad.detach().float()
    q_left, q_right = state["Q"]
    inv_left, inv_right = state["eigen_sqrt_inv"]
    beta = state["shampoo_beta"]
    rows, cols = grad.shape

    right_whitened = (q_right.T @ grad.T) * inv_right.view(-1, 1)
    left_target = right_whitened.T @ right_whitened / cols
    left_whitened = (q_left.T @ grad) * inv_left.view(-1, 1)
    right_target = left_whitened.T @ left_whitened / rows
    state["GG"][0].mul_(beta).add_(left_target, alpha=1.0 - beta)
    state["GG"][1].mul_(beta).add_(right_target, alpha=1.0 - beta)
    state["GG"][0] = _symmetrize(state["GG"][0]).contiguous()
    state["GG"][1] = _symmetrize(state["GG"][1]).contiguous()

    projected = q_left.T @ grad @ q_right
    left_diag = (projected * inv_right.view(1, -1)).square().mean(dim=1)
    right_diag = (projected * inv_left.view(-1, 1)).square().mean(dim=0)
    _update_eigen_sqrt_inv_(state, left_diag, 0, beta)
    _update_eigen_sqrt_inv_(state, right_diag, 1, beta)


@torch.no_grad()
def refresh_klsoap_basis_(state):
    exp_avg_original = project_from_klsoap_basis(state["exp_avg"], state)
    refreshed = []
    for gg, q in zip(state["GG"], state["Q"]):
        new_q, _ = torch.linalg.qr(gg.float() @ q.float())
        refreshed.append(new_q.contiguous())
    state["Q"] = refreshed
    state["exp_avg"] = project_to_klsoap_basis(exp_avg_original, state).contiguous()


def klsoap_direction(state, grad: Tensor, beta1: float, beta2: float, eps: float) -> Tensor:
    grad_projected = project_to_klsoap_basis(grad.detach().float(), state)
    state["exp_avg"].mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)
    state["exp_avg_sq"].mul_(beta2).addcmul_(grad_projected, grad_projected, value=1.0 - beta2)
    preconditioned = state["exp_avg"] / (state["exp_avg_sq"].sqrt() + eps)
    return project_from_klsoap_basis(preconditioned, state)


class KLSOAPH(torch.optim.Optimizer):
    def __init__(
        self, params, lr=0.018, beta1=0.95, beta2=0.9,
        shampoo_beta=0.9, eps=1e-8, precondition_frequency=1,
    ):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, shampoo_beta=shampoo_beta,
            eps=eps, precondition_frequency=precondition_frequency,
        )
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
                        init_2d_klsoap_state_(state, p.grad, group["shampoo_beta"], init_factor=0.1)
                        dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
                        continue
                    state["step"] += 1
                    update = klsoap_direction(state, p.grad, group["beta1"], group["beta2"], group["eps"])
                    update_2d_klsoap_preconditioner_(p.grad, state)
                    if state["step"] % group["precondition_frequency"] == 0:
                        refresh_klsoap_basis_(state)
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

train_seed = int(os.environ.get("KL_SOAP_SEED", "1"))
torch.manual_seed(train_seed)
torch.cuda.manual_seed_all(train_seed)
print0(f"Using seed={train_seed}", console=True)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


num_trials = 1

for _ in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = 3040
    schedule_steps = 3125
    lr_power = 1.5
    adam_lr_floor = 0.04
    klsoap_lr_floor = 0.017
    print0("KL-SOAP-H config: train_steps=3040 schedule_steps=3125 lr_power=1.5 "
           "adam_lr_floor=0.04 klsoap_lr_floor=0.017 "
           "beta1=0.95 beta2=0.9 shampoo_beta=0.9 lr=0.018 precondition_frequency=1", console=True)

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
                  if n.endswith(".attn.q.weight") or n.endswith(".attn.k.weight") or n.endswith(".attn.v.weight")]
    mlp_fc_params = [p for n, p in named_block_params if n.endswith(".mlp.fc.weight")]
    attn_proj_params = [p for n, p in named_block_params if n.endswith(".attn.proj.weight")]
    mlp_proj_params = [p for n, p in named_block_params if n.endswith(".mlp.proj.weight")]

    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                        dict(params=[model.proj.weight], lr=1/320),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
    optimizer2 = KLSOAPH(qkv_params, lr=0.018)
    optimizer3 = KLSOAPH(mlp_fc_params, lr=0.018)
    optimizer4 = KLSOAPH(attn_proj_params, lr=0.018)
    optimizer5 = KLSOAPH(mlp_proj_params, lr=0.018)
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
            group["lr_floor"] = klsoap_lr_floor
    for group in optimizer1.param_groups:
        group["cooldown_frac"] = 0.4
        group["lr_floor"] = adam_lr_floor

    def set_hparams(step):
        progress = step / schedule_steps
        assert 0 <= progress < 1
        for opt in optimizers:
            for group in opt.param_groups:
                cooldown_frac = group["cooldown_frac"]
                if progress < 1 - cooldown_frac:
                    eta = 1.0
                else:
                    x = (1 - progress) / cooldown_frac
                    eta = group["lr_floor"] + (1 - group["lr_floor"]) * x ** lr_power
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
        val_step_freq = 125 if step / train_steps < 0.9 else 25
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
