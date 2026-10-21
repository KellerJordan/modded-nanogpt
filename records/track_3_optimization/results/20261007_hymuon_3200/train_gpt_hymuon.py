"""HyMUON (Hybrid Momentum MUON), Track 3 adaptation, 3,200 updates.

Retains the paper's incremental full-history recurrence; fast EMA, mixing,
Nesterov blend, NS12, AdamW auxiliaries, and raw evaluation are adaptations.
The archived source and exact recorded launch settings are embedded in logs/.
"""
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import argparse
import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


def tensor_collection_sha256(named_tensors):
    """Hash names, dtype, shape, and complete tensor bytes in stable order."""
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


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

def zeropower_via_newtonschulz(G: Tensor, ns_steps: int) -> Tensor:
    """Baseline quintic NS polynomial, with a preregistered iteration count.

    The locked baseline uses (2, -1.5, 0.5), BF16 iterates, transpose-on-tall,
    Frobenius normalization with 1e-7, and 12 iterations.  The paper states
    five quintic iterations but not their coefficients; ns_steps=5 therefore
    means a declared five-iteration reuse of this common baseline polynomial,
    not a claim of bitwise parity with the authors' unpublished kernel.
    """
    assert ns_steps in (5, 12)
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True, ns_steps=12):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz(update, ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

@torch.compile
def nested_muon_update(grad, momentum, slow, mass, step, *, alpha, mu=0.95, ns_steps=12):
    """CleanMuon NAG with a triangular full-history mean nested in fast memory."""
    momentum.lerp_(grad, 1 - mu)
    # step is the current one-indexed global optimizer update and includes grad.
    slow.mul_((step - 1.0) / (step + 1.0)).add_(grad, alpha=2.0 / (step + 1.0))
    mixed_memory = momentum.float().mul(1.0 - alpha).add_(slow, alpha=alpha * mass)
    direction = grad.float().mul(1.0 - mu).add_(mixed_memory, alpha=mu)
    update = zeropower_via_newtonschulz(direction, ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

@torch.compile
def hymuon_matrix_update(grad, slow, fast, step, *, mu=0.95, beta=0.99, ns_steps=12):
    """Appendix F.7 equations (74)-(77), with no extra Nesterov term."""
    step.add_(1.0)
    c_t = step.reciprocal()
    slow.mul_(1.0 - c_t).add_(grad)
    fast.mul_(mu * (1.0 - c_t)).add_(grad)
    q_t = (1.0 - torch.pow(mu, step)) / (1.0 - mu)
    direction = torch.add(slow * (beta * c_t * q_t), fast, alpha=1.0 - beta)
    # Algebraically equation (76) is exactly G_1.  Preserve that identity
    # numerically before BF16 NS instead of amplifying scalar roundoff.
    direction = torch.where(step == 1.0, grad, direction)
    update = zeropower_via_newtonschulz(direction, ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

@torch.compile
def dual_hym_update(grad, slow1, fast1, slow2, fast2, step, *,
                    beta=0.99, beta2=0.95, eps=1e-10):
    """Appendix F.7 equations (78)-(81), without Adam bias correction."""
    step.add_(1.0)
    c_t = step.reciprocal()
    grad2 = grad.square()
    slow1.mul_(1.0 - c_t).add_(grad)
    fast1.mul_(beta * (1.0 - c_t)).add_(grad)
    slow2.mul_(1.0 - c_t).add_(grad2)
    fast2.mul_(beta2 * (1.0 - c_t)).add_(grad2)
    q1 = (1.0 - torch.pow(beta, step)) / (1.0 - beta)
    q2 = (1.0 - torch.pow(beta2, step)) / (1.0 - beta2)
    d1 = torch.add(slow1 * (beta * c_t * q1), fast1, alpha=1.0 - beta)
    # Paper Eq. (80) deliberately keeps outer beta for both r=1 and r=2;
    # rho_2=beta2 appears in b^(2) and q_t(rho_2), not in the outer mixture.
    d2 = torch.add(slow2 * (beta * c_t * q2), fast2, alpha=1.0 - beta)
    d1 = torch.where(step == 1.0, grad, d1)
    d2 = torch.where(step == 1.0, grad2, d2)
    # Appendix F.7 equation (81): epsilon is inside the square root.
    return d1 / torch.sqrt(d2 + eps)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, ns_steps=12,
                 nested_alpha=0.0):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        assert nested_alpha in (0.0, 0.01, 0.025, 0.05, 0.10)
        # Keep the alpha=0 optimizer state_dict identical to canonical CleanMuon:
        # do not even add inert param-group fields in the control branch.
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.nested_alpha = float(nested_alpha)
        if self.nested_alpha != 0.0:
            for group in self.param_groups:
                group["nested_alpha"] = self.nested_alpha
                group["nested_global_step"] = 0
                group["nested_fast_mass"] = 0.0

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            alpha = self.nested_alpha
            if alpha != 0.0:
                # Exactly once per optimizer update, not once per parameter.
                group["nested_global_step"] += 1
                group["nested_fast_mass"] = (
                    group["mu"] * group["nested_fast_mass"] + (1.0 - group["mu"])
                )
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                        if alpha != 0.0:
                            state["nested_slow"] = torch.zeros_like(p, dtype=torch.float32)
                    if alpha == 0.0:
                        # Preserve the canonical operation order bit-for-bit.
                        update = muon_update(p.grad, state["momentum"], mu=group["mu"],
                                             ns_steps=group["ns_steps"])
                    else:
                        update = nested_muon_update(
                            p.grad, state["momentum"], state["nested_slow"],
                            group["nested_fast_mass"], group["nested_global_step"],
                            alpha=alpha, mu=group["mu"], ns_steps=group["ns_steps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

class HyMuon(torch.optim.Optimizer):
    """Distributed hidden-matrix HyM -> NS optimizer (paper eqs. 74-77, 83)."""
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95,
                 beta=0.99, ns_steps=12):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu,
                        beta=beta, ns_steps=ns_steps)
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
                        # Appendix F.7 does not fix state dtype.  FP32 is a
                        # frozen Track-3 reimplementation choice.
                        state["slow"] = torch.zeros_like(p, dtype=torch.float32)
                        state["fast"] = torch.zeros_like(p, dtype=torch.float32)
                        state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    grad = p.grad.to(torch.float32)
                    update = hymuon_matrix_update(
                        grad, state["slow"], state["fast"], state["step"],
                        mu=group["mu"], beta=group["beta"], ns_steps=group["ns_steps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.to(p.dtype), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

class DualHyM(torch.optim.Optimizer):
    """Appendix F.7 dual-HyM adaptive auxiliary branch (eqs. 78-81, 84)."""
    def __init__(self, param_groups, beta=0.99, beta2=0.95,
                 eps=1e-10, weight_decay=0.001):
        defaults = dict(beta=beta, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta, beta2, eps = group["beta"], group["beta2"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # FP32 states are frozen and disclosed because the paper
                    # does not specify optimizer-state dtype for this setting.
                    state["slow1"] = torch.zeros_like(p, dtype=torch.float32)
                    state["fast1"] = torch.zeros_like(p, dtype=torch.float32)
                    state["slow2"] = torch.zeros_like(p, dtype=torch.float32)
                    state["fast2"] = torch.zeros_like(p, dtype=torch.float32)
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                grad = p.grad.to(torch.float32)
                update = dual_hym_update(
                    grad, state["slow1"], state["fast1"],
                    state["slow2"], state["fast2"], state["step"],
                    beta=beta, beta2=beta2, eps=eps)
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"])

class OnlineXPoint:
    """Read-only paper equation (82), advanced once before each parameter update."""
    def __init__(self, params, beta=0.99):
        self.params = list(params)
        self.beta = float(beta)
        self.step = 0
        # The paper does not specify x-point state dtype.  FP32 is frozen here.
        self.x = [p.detach().to(torch.float32).clone() for p in self.params]

    @torch.no_grad()
    def capture_preupdate(self):
        self.step += 1
        c_t = 1.0 / self.step
        denom = (1.0 - c_t) * (1.0 - self.beta) + c_t
        old_weight = (1.0 - c_t) * (1.0 - self.beta) / denom
        new_weight = c_t / denom
        for x, p in zip(self.x, self.params):
            x.mul_(old_weight).add_(p.detach().to(torch.float32), alpha=new_weight)

    @contextmanager
    def use_xpoint(self):
        with torch.no_grad():
            raw = [p.detach().clone() for p in self.params]
            try:
                for p, x in zip(self.params, self.x):
                    p.copy_(x.to(p.dtype))
                yield
            finally:
                for p, value in zip(self.params, raw):
                    p.copy_(value)

    def state_dict(self):
        return {"beta": self.beta, "step": self.step,
                "x": [value.detach().clone() for value in self.x]}

    def load_state_dict(self, state):
        assert float(state["beta"]) == self.beta
        assert len(state["x"]) == len(self.x)
        self.step = int(state["step"])
        for target, source in zip(self.x, state["x"]):
            target.copy_(source)


########################################
#                Setup                 #
########################################

ARM_CONFIGS = {
    # Canonical Track-3 baseline, with passive x retained only as a control.
    "muon12": dict(matrix="muon", aux="adamw", ns_steps=12, primary="raw"),
    # Full HyMuon: matrix HyM -> NS, dual-HyM aux, x primary.
    "hymuon12": dict(matrix="hymuon", aux="dual_hym", ns_steps=12, primary="x"),
}

INCUMBENT_CONFIGS = {
    "muon12": dict(
        provenance="method_anchor_dev352_353_selected_global_lrx1",
        prior_global_lr_multiplier=1.0,
        matrix_lr=0.025, matrix_weight_decay=0.05, matrix_mu=0.95,
        hym_beta=None,
        aux_lrs=dict(embedding=0.7, head=0.004, one_dimensional=0.015),
        aux_weight_decay=0.001, aux_beta1=0.8, aux_beta2=0.95,
        aux_eps=1e-10),
    "hymuon12": dict(
        provenance="method_anchor_dev352_353_selected_global_lrx2",
        prior_global_lr_multiplier=2.0,
        matrix_lr=0.02, matrix_weight_decay=0.2, matrix_mu=0.95,
        hym_beta=0.99,
        aux_lrs=dict(embedding=0.01, head=0.01, one_dimensional=0.01),
        aux_weight_decay=0.1, aux_beta1=0.99, aux_beta2=0.999,
        aux_eps=1e-8),
}

LR_CONFIGS = {
    "lr025": dict(matrix_multiplier=1.0, aux_multiplier=1.0, matrix_peak_lr=0.025, matrix_weight_decay=0.05),
    "lr030": dict(matrix_multiplier=0.030/0.025, aux_multiplier=1.0, matrix_peak_lr=0.030, matrix_weight_decay=1.0/24.0),
}


WD_CONFIGS = {"wd050":0.05,"wd100":0.10}
NESTED_CONFIRM_SEEDS = (636, 637, 638, 639, 640, 641, 642, 643)
SLOW_DECAYS = (0.98,)


def config_tag(alpha, slow_decay, lr_config, wd_config):
    return "tri-a" + str(int(round(alpha*1000))).zfill(3) + "-" + lr_config + "-" + wd_config

def expected_formal_run_id(seed, steps, alpha, slow_decay, lr_config, wd_config):
    return f"hymcoreconfirm-seed{seed}-s{steps}-{config_tag(alpha, slow_decay, lr_config, wd_config)}"

def expected_gate_run_id(steps, alpha, slow_decay, lr_config, wd_config):
    return f"hymcoreconfirm-gate221-s{steps}-{config_tag(alpha, slow_decay, lr_config, wd_config)}"

def validate_run_admission(args):
    assert args.formal != args.integration_gate, "exactly one declared mode required"
    assert args.train_steps == 3200
    assert args.arm == "muon12"
    assert args.lr_config == "lr030" and args.wd_config == "wd050"
    assert args.nested_alpha == 0.05 and args.slow_decay is None
    if args.integration_gate:
        assert args.seed == 221
        assert args.run_id == expected_gate_run_id(args.train_steps,args.nested_alpha,None,"lr030","wd050")
        return
    release_path = os.environ.get("HYM_RELEASE_FILE", str(Path(__file__).with_name("release_manifest.json")))
    release_hash = os.environ.get("HYM_RELEASE_SHA256")
    if release_hash is None and "HYM_RELEASE_FILE" not in os.environ:
        release_hash = hashlib.sha256(Path(release_path).read_bytes()).hexdigest()
    assert release_path and release_hash, "development admission closed without hash-bound manifest"
    blob = Path(release_path).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == release_hash
    r = json.loads(blob)
    assert r["schema"] == "hym_core_3200_fresh_confirmation_v1"
    assert r["trainer_sha256"] == hashlib.sha256(code.encode()).hexdigest()
    assert r["data_manifest_sha256"] == "5b11f0f86b7d07f436ccfe2ac30346ccc2c79537b031ef19e0006359db90b30c"
    assert r["confirmation_seeds"] == list(NESTED_CONFIRM_SEEDS) and args.seed in r["confirmation_seeds"]
    assert r["steps"] == [3200]
    assert {"steps":args.train_steps,"alpha":args.nested_alpha,"lr":"lr030","wd":"wd050"} in r["configs"]
    assert args.run_id == expected_formal_run_id(args.seed,args.train_steps,args.nested_alpha,None,"lr030","wd050")

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="muon12", choices=sorted(ARM_CONFIGS))
parser.add_argument("--seed", default=636, type=int)
parser.add_argument("--train-steps", type=int, default=3200)
parser.add_argument("--lr-config", default="lr030", choices=sorted(LR_CONFIGS))
parser.add_argument("--wd-config", default="wd050", choices=sorted(WD_CONFIGS))
parser.add_argument("--run-id")
parser.add_argument("--formal", action="store_true")
parser.add_argument("--integration-gate", action="store_true")
parser.add_argument("--admission-only", action="store_true")
parser.add_argument("--nested-alpha", type=float, choices=(0.01, 0.025, 0.05, 0.10), default=0.05)
parser.add_argument("--slow-decay", type=float, choices=SLOW_DECAYS)
args = parser.parse_args()
if not args.formal and not args.integration_gate:
    args.formal = True
if args.run_id is None:
    args.run_id = (expected_gate_run_id(args.train_steps, args.nested_alpha, args.slow_decay, args.lr_config, args.wd_config)
                   if args.integration_gate else
                   expected_formal_run_id(args.seed, args.train_steps, args.nested_alpha, args.slow_decay, args.lr_config, args.wd_config))
assert args.train_steps > 0
assert not (args.formal and args.integration_gate)
arm_config = ARM_CONFIGS[args.arm]
recipe_config = dict(INCUMBENT_CONFIGS[args.arm])
lr_config = LR_CONFIGS[args.lr_config]
validate_run_admission(args)
if args.admission_only:
    print("ADMISSION_ONLY_JSON " + json.dumps({"status":"PASS","mode":"formal_schema" if args.formal else "technical", "no_model_data_GPU_or_seed_initialization":True},sort_keys=True))
    raise SystemExit(0)
run_updates = 16 if args.integration_gate else args.train_steps

matrix_peak_lr = lr_config["matrix_peak_lr"]
recipe_config["matrix_weight_decay"] = WD_CONFIGS[args.wd_config]
aux_peak_lrs = {
    name: value * lr_config["aux_multiplier"]
    for name, value in recipe_config["aux_lrs"].items()
}

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
print0("PREREG_CONFIG " + json.dumps({
    "run_id": args.run_id,
    "formal": args.formal,
    "integration_gate": args.integration_gate,
    "arm": args.arm,
    "seed": args.seed,
    "train_steps": args.train_steps,
    "run_updates": run_updates,
    "study": "HyMUON_Track3_fresh_confirmation636643",
    "incumbent_provenance": recipe_config["provenance"],
    "prior_global_lr_multiplier": recipe_config["prior_global_lr_multiplier"],
    "lr_config": args.lr_config,
    "wd_config": args.wd_config,
    "matrix_peak_weight_decay": recipe_config["matrix_weight_decay"],
    "matrix_wd_schedule": "peakWD_times_eta__matrix_LRWD_product_peakLR_peakWD_eta_squared",
    "matrix_lr_multiplier_from_incumbent": lr_config["matrix_multiplier"],
    "aux_lr_multiplier_from_incumbent": lr_config["aux_multiplier"],
    "matrix": arm_config["matrix"],
    "aux": arm_config["aux"],
    "ns_steps": arm_config["ns_steps"],
    "primary_readout": arm_config["primary"],
    "primary_parse_key": "primary_val_loss",
    "trainer_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
    "matrix_incumbent_peak_lr": recipe_config["matrix_lr"],
    "matrix_effective_peak_lr": matrix_peak_lr,
    "matrix_weight_decay": recipe_config["matrix_weight_decay"],
    "matrix_peak_lr_times_weight_decay": matrix_peak_lr * recipe_config["matrix_weight_decay"],
    "x_beta": 0.99,
    "x_update_point": "once_pre_optimizer_after_gradient_allreduce",
    "hym_beta": recipe_config["hym_beta"],
    "matrix_mu": recipe_config["matrix_mu"],
    "nested_alpha": args.nested_alpha,
    "slow_decay": args.slow_decay,
    "nested_slow_recurrence": "S_t=(t-1)/(t+1)*S_(t-1)+2*G_t/(t+1); S_0=0; full_history_weights=2i/[t(t+1)]",
    "nested_fast_mass_recurrence": "w_t=mu*w_(t-1)+(1-mu);w_0=0",
    "nested_direction": "(1-mu)*G+mu*((1-alpha)*m+alpha*w*S)",
    "aux_incumbent_peak_lrs": recipe_config["aux_lrs"],
    "aux_effective_peak_lrs": aux_peak_lrs,
    "aux_weight_decay": recipe_config["aux_weight_decay"],
    "aux_peak_lr_times_weight_decay": {
        name: value * recipe_config["aux_weight_decay"]
        for name, value in aux_peak_lrs.items()
    },
    "aux_beta1": recipe_config["aux_beta1"],
    "aux_beta2": recipe_config["aux_beta2"],
    "aux_eps": recipe_config["aux_eps"],
    "dual_hym_epsilon_placement": "inactive_legacy_inside_sqrt_variant__actual_paper_eq81_is_outside",
    "muon_matrix_state_dtype": "parameter_dtype__fp32_for_hidden_matrices",
    "muon_aux_moment_dtype": "parameter_dtype__bf16_embedding__fp32_head_and_1d",
    "hymuon_matrix_aux_x_state_dtype": "fp32",
    "lr_schedule": "30pct_flat_then_70pct_linear",
    "global_gradient_clip": None,
    "ns_polynomial": [2.0, -1.5, 0.5],
    "ns_normalization_epsilon": 1e-7,
    "ns_iterate_dtype": "bfloat16",
    "ns_transpose_if_rows_gt_columns": True,
    "ns_shape_factor": "sqrt(max(1,m/n))_after_ns",
    "task_adaptations": [
        "track3_model_data_global_batch_and_3200_update_endpoint",
        "track3_ns12_for_both_methods__paper_table6_uses_five_ns_iterations",
        "no_global_clip_for_both_methods",
        "track3_30pct_flat_70pct_linear_schedule__no_table6_2000_step_warmup_or_cosine",
        "track3_raw_only_readout_for_nested_cleanmuon_development",
    ],
    "confirmation_budget": {
        "configs": ["s3200-tri-a050-lr030-wd050"],
        "confirmation_seeds": list(NESTED_CONFIRM_SEEDS),
        "seeds_per_configuration": 8,
        "total_runs": 8,
    },
    "historical_tuning_budget_claimed_equal": False,
    "finalist_source": "frozen_required_incremental_history_core_20261006",
    "selection_rule": "all8_fresh_confirmation__official_sqrt_n_times_3.28_minus_mean_ge_0.004__no_rescue_no_retry",
    "selection_endpoint": "fixed_step_3200_raw_primary",
    "seed_exclusion": "fresh636643_only__development658659_excluded",
    "baseline_commit": "4ea6b937337a4889b8cfe3f38a93d120048d8f71",
}, sort_keys=True))
print0("LAUNCH_ARGS_JSON " + json.dumps({"argv": sys.argv[1:]}, sort_keys=True))
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))
if dist.get_rank() == 0:
    print0("VAL_DATA_HASH_JSON " + json.dumps({
        "data_order": "sorted_shards_sequential_no_rng",
        "rank": 0,
        "world_size": dist.get_world_size(),
        "val_tokens": val_tokens,
        "sha256": tensor_collection_sha256([
            ("val_inputs", val_inputs), ("val_targets", val_targets)]),
    }, sort_keys=True))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
assert sum(p.numel() for p in model.parameters()) == 162_354_816
model.compile(dynamic=False)


num_trials = 1

for _ in range(num_trials):


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = args.train_steps

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    aux_groups = [dict(params=[model.embed.weight], lr=aux_peak_lrs["embedding"]),
                  dict(params=[model.proj.weight], lr=aux_peak_lrs["head"]),
                  dict(params=[p for p in model.parameters() if p.ndim < 2],
                       lr=aux_peak_lrs["one_dimensional"])]
    if arm_config["aux"] == "adamw":
        optimizer1 = AdamW(
            aux_groups,
            betas=(recipe_config["aux_beta1"], recipe_config["aux_beta2"]),
            eps=recipe_config["aux_eps"],
            weight_decay=recipe_config["aux_weight_decay"], fused=True)
    else:
        optimizer1 = DualHyM(
            aux_groups, beta=recipe_config["aux_beta1"],
            beta2=recipe_config["aux_beta2"], eps=recipe_config["aux_eps"],
            weight_decay=recipe_config["aux_weight_decay"])
    matrix_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    if arm_config["matrix"] == "muon":
        optimizer2 = Muon(
            matrix_params, lr=matrix_peak_lr,
            weight_decay=recipe_config["matrix_weight_decay"],
            mu=recipe_config["matrix_mu"], ns_steps=arm_config["ns_steps"],
            nested_alpha=args.nested_alpha)
    else:
        optimizer2 = HyMuon(
            matrix_params, lr=matrix_peak_lr,
            weight_decay=recipe_config["matrix_weight_decay"],
            mu=recipe_config["matrix_mu"], beta=recipe_config["hym_beta"],
            ns_steps=arm_config["ns_steps"])
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    for group in optimizer2.param_groups:
        group["initial_weight_decay"] = group["weight_decay"]

    @torch.no_grad()
    def snapshot_optimizer_groups(optimizer):
        """Snapshot only at preregistered audit updates; no training feedback."""
        return [
            [(p, p.detach().to(torch.float32).clone())
             for p in group["params"] if p.grad is not None]
            for group in optimizer.param_groups
        ]

    @torch.no_grad()
    def summarize_optimizer_update(optimizer, snapshots, role):
        summaries = []
        for group_index, (group, entries) in enumerate(
                zip(optimizer.param_groups, snapshots)):
            parameter_sq = 0.0
            gradient_sq = 0.0
            delta_sq = 0.0
            for parameter, before in entries:
                parameter_sq += float(before.square().sum())
                gradient_sq += float(parameter.grad.detach().to(torch.float32).square().sum())
                delta_sq += float(
                    (parameter.detach().to(torch.float32) - before).square().sum())
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            summaries.append({
                "role": role,
                "group_index": group_index,
                "parameter_tensors": len(entries),
                "lr": lr,
                "weight_decay": weight_decay,
                "lr_times_weight_decay": lr * weight_decay,
                "parameter_l2_before": parameter_sq**0.5,
                "gradient_l2_before": gradient_sq**0.5,
                "actual_parameter_delta_l2": delta_sq**0.5,
            })
        return summaries

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
        for group in optimizer2.param_groups:
            group["weight_decay"] = group["initial_weight_decay"] * eta


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    if dist.get_rank() == 0:
        print0("FULL_INIT_RNG_HASH_JSON " + json.dumps({
            "seed": args.seed,
            "model_sha256": tensor_collection_sha256(model.named_parameters()),
            "torch_cpu_rng_sha256": tensor_collection_sha256(
                [("torch_cpu_rng", torch.get_rng_state())]),
            "torch_cuda_rng_sha256": tensor_collection_sha256(
                [("torch_cuda_rng", torch.cuda.get_rng_state(device))]),
            "data_rng": "none__sorted_shards_sequential_loader",
        }, sort_keys=True), console=True)
    import platform, triton
    print0("ACTUAL_RUNTIME_JSON " + json.dumps({
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "triton": triton.__version__,
        "world_size": dist.get_world_size(), "seed": args.seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_SHA256": hashlib.sha256(code.encode()).hexdigest(),
        "parameter_dtype_counts": {str(d): sum(p.numel() for p in model.parameters() if p.dtype==d) for d in {p.dtype for p in model.parameters()}},
    }, sort_keys=True), console=True)
    xpoint = OnlineXPoint(model.parameters(), beta=0.99)

    def evaluate_current_parameters():
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        return float(val_loss / val_tokens)

    # start the clock
    training_time = 0
    last_val_step = 0
    integration_checkpoint_verified = False
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(run_updates + 1):

        # --------------- VALIDATION SECTION -----------------
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == run_updates or step % val_step_freq == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            assert xpoint.step == step, (xpoint.step, step)
            if step == run_updates:
                val_loss_raw = evaluate_current_parameters()
                raw_model_sha_before_x = None
                if dist.get_rank() == 0:
                    raw_model_sha_before_x = tensor_collection_sha256(model.named_parameters())
                # The nested CleanMuon study has one frozen raw endpoint; the
                # passive x state is inherited but is never used as a readout.
                if arm_config["matrix"] == "muon":
                    val_loss_x = None
                else:
                    with xpoint.use_xpoint():
                        val_loss_x = evaluate_current_parameters()
                raw_model_sha_after_x = None
                if dist.get_rank() == 0:
                    raw_model_sha_after_x = tensor_collection_sha256(model.named_parameters())
                    assert raw_model_sha_after_x == raw_model_sha_before_x, (
                        raw_model_sha_before_x, raw_model_sha_after_x)
                primary_readout = arm_config["primary"]
                primary_val_loss = val_loss_raw if primary_readout == "raw" else val_loss_x
                x_text = "none" if val_loss_x is None else f"{val_loss_x:.5f}"
                print0(f"step:{step}/{run_updates} val_loss:{primary_val_loss:.5f}"
                       + f" val_loss_raw:{val_loss_raw:.5f} val_loss_x:{x_text}"
                       + f" readout:{primary_readout} train_time:{training_time:.3f}s"
                       + f" step_avg:{1000*step_avg:.2f}ms", console=True)
                print0("FINAL_METRICS_JSON " + json.dumps({
                    "run_id": args.run_id,
                    "arm": args.arm,
                    "seed": args.seed,
                    "study": "HyMUON_Track3_fresh_confirmation636643",
                    "nested_alpha": args.nested_alpha,
                    "slow_decay": args.slow_decay,
                    "lr_config": args.lr_config,
    "wd_config": args.wd_config,
    "matrix_peak_weight_decay": recipe_config["matrix_weight_decay"],
    "matrix_wd_schedule": "peakWD_times_eta__matrix_LRWD_product_peakLR_peakWD_eta_squared",
                    "matrix_lr_multiplier_from_incumbent": lr_config["matrix_multiplier"],
                    "aux_lr_multiplier_from_incumbent": lr_config["aux_multiplier"],
                    "trainer_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    "step": step,
                    "updates": step,
                    "configured_schedule_horizon": train_steps,
                    "integration_gate": args.integration_gate,
                    "integration_checkpoint_verified": integration_checkpoint_verified,
                    "xpoint_last_included_preupdate": xpoint.step,
                    "val_loss_raw": val_loss_raw,
                    "val_loss_x": val_loss_x,
                    "primary_readout": primary_readout,
                    "primary_val_loss": primary_val_loss,
                    "eval_restore_verified": raw_model_sha_after_x == raw_model_sha_before_x,
                    "raw_model_sha256": raw_model_sha_after_x,
                    "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(device),
                    "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "nested_slow_state_bytes": sum(
                        value.numel() * value.element_size()
                        for state in optimizer2.state.values()
                        for key, value in state.items() if key == "nested_slow"),
                }, sort_keys=True), console=True)
                print0(f"PRIMARY_ENDPOINT run_id={args.run_id} arm={args.arm}"
                       + f" seed={args.seed} lr_config={args.lr_config} step={step}"
                       + f" readout={primary_readout} primary_val_loss={primary_val_loss:.8f}",
                       console=True)
            else:
                primary_readout = arm_config["primary"]
                if primary_readout == "raw":
                    val_loss = evaluate_current_parameters()
                else:
                    with xpoint.use_xpoint():
                        val_loss = evaluate_current_parameters()
                print0(f"step:{step}/{run_updates} val_loss:{val_loss:.5f}"
                       + f" readout:{primary_readout} train_time:{training_time:.3f}s"
                       + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == run_updates:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        if step == 0 and dist.get_rank() == 0:
            print0("FIRST_TRAIN_BATCH_HASH_JSON " + json.dumps({
                "seed_independent": True,
                "rank": 0,
                "world_size": dist.get_world_size(),
                "global_tokens": batch_size,
                "local_tokens": inputs.numel(),
                "sha256": tensor_collection_sha256(
                    [("train_inputs", inputs), ("train_targets", targets)]),
            }, sort_keys=True), console=True)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        # set optimization hyperparameters and take a step
        set_hparams(step)
        # Eq. (82) includes theta_t before equations (83)-(84) produce theta_(t+1).
        xpoint.capture_preupdate()
        # Record actual LR/WD products and full parameter-delta norms at the
        # first and final updates.  Snapshots are diagnostics only and never
        # enter optimizer state or alter the update.
        collect_update_audit = step in (0, run_updates - 1)
        snapshots = ([snapshot_optimizer_groups(opt) for opt in optimizers]
                     if collect_update_audit else None)
        for opt in optimizers:
            opt.step()
        if args.integration_gate and step == 7:
            # Exercise a real optimizer checkpoint round-trip at update 8 while
            # retaining the same model/data stream. In particular this saves
            # and restores per-parameter S plus group-level t and w.
            cpu_rng_before_checkpoint = torch.get_rng_state().clone()
            cuda_rng_before_checkpoint = torch.cuda.get_rng_state(device).clone()
            before_group = {
                key: optimizer2.param_groups[0].get(key)
                for key in ("nested_global_step", "nested_fast_mass")}
            before_slow_hash = tensor_collection_sha256([
                (str(index), state["nested_slow"])
                for index, state in enumerate(optimizer2.state.values())
                if "nested_slow" in state])
            checkpoint_buffer = io.BytesIO()
            torch.save(optimizer2.state_dict(), checkpoint_buffer)
            checkpoint_buffer.seek(0)
            optimizer2.load_state_dict(torch.load(
                checkpoint_buffer, map_location=device, weights_only=False))
            after_group = {
                key: optimizer2.param_groups[0].get(key)
                for key in ("nested_global_step", "nested_fast_mass")}
            after_slow_hash = tensor_collection_sha256([
                (str(index), state["nested_slow"])
                for index, state in enumerate(optimizer2.state.values())
                if "nested_slow" in state])
            assert before_group == after_group
            assert before_slow_hash == after_slow_hash
            assert torch.equal(cpu_rng_before_checkpoint, torch.get_rng_state())
            assert torch.equal(cuda_rng_before_checkpoint, torch.cuda.get_rng_state(device))
            if args.nested_alpha != 0.0:
                assert after_group["nested_global_step"] == 8
            integration_checkpoint_verified = True
            print0("INTEGRATION_CHECKPOINT_JSON " + json.dumps({
                "completed_update": 8,
                "nested_alpha": args.nested_alpha,
                "group_state": after_group,
                "nested_slow_sha256": after_slow_hash,
                "rng_unchanged": True,
            }, sort_keys=True), console=True)
        if collect_update_audit:
            update_audit = []
            for role, opt, before in zip(
                    ("auxiliary", "matrix"), optimizers, snapshots):
                update_audit.extend(summarize_optimizer_update(opt, before, role))
            print0("OPTIMIZER_UPDATE_AUDIT_JSON " + json.dumps({
                "run_id": args.run_id,
                "arm": args.arm,
                "seed": args.seed,
                "lr_config": args.lr_config,
    "wd_config": args.wd_config,
    "matrix_peak_weight_decay": recipe_config["matrix_weight_decay"],
    "matrix_wd_schedule": "peakWD_times_eta__matrix_LRWD_product_peakLR_peakWD_eta_squared",
                "completed_update": step + 1,
                "groups": update_audit,
            }, sort_keys=True), console=True)
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{run_updates} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

dist.destroy_process_group()
