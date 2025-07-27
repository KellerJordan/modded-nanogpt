import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache, partial # Added partial for hook registration
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=0.12).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean')
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48*1024 # FlexAttention sequence length
    val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    # optimization
    num_iterations = 1770 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
args = Hyperparameters()

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 8 # this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
adam_params = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

model: nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

########################################
#      Overlap Communication Setup     #
########################################

# Create parameter buckets for better overlap
def create_buckets(params, bucket_size_mb=25):
    """Group parameters into buckets of approximately bucket_size_mb MB each"""
    buckets = []
    current_bucket = []
    current_size = 0

    # Sort parameters by size (largest first) for better bucketing
    sorted_params = sorted(params, key=lambda p: p.numel(), reverse=True)

    for param in sorted_params:
        param_size_mb = param.numel() * param.element_size() / (1024 * 1024)

        if current_size + param_size_mb > bucket_size_mb and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [param]
            current_size = param_size_mb
        else:
            current_bucket.append(param)
            current_size += param_size_mb

    if current_bucket:
        buckets.append(current_bucket)

    return buckets

# Create buckets for all parameters
all_params = [p for p in model.parameters() if p.requires_grad]
param_buckets = create_buckets(all_params)

print0(f"Created {len(param_buckets)} gradient buckets")
for i, bucket in enumerate(param_buckets):
    total_size = sum(p.numel() * p.element_size() for p in bucket) / (1024 * 1024)
    print0(f"Bucket {i}: {len(bucket)} params, {total_size:.1f} MB")

# Bucket state tracking
bucket_ready_count = [0] * len(param_buckets)
bucket_handles = [None] * len(param_buckets)
param_to_bucket = {}

# Map each parameter to its bucket index
for bucket_idx, bucket in enumerate(param_buckets):
    for param in bucket:
        param_to_bucket[param] = bucket_idx

def _gradient_hook(param: Tensor):
    """Called when a parameter's gradient is ready"""
    if param.grad is None:
        return

    bucket_idx = param_to_bucket[param]
    bucket_ready_count[bucket_idx] += 1

    # Check if all parameters in this bucket are ready
    if bucket_ready_count[bucket_idx] == len(param_buckets[bucket_idx]):
        # All-reduce this bucket
        bucket_grads = [p.grad for p in param_buckets[bucket_idx]]

        # For multi-tensor operations, we can reduce them together
        if len(bucket_grads) == 1:
            handle = dist.all_reduce(bucket_grads[0], op=dist.ReduceOp.AVG, async_op=True)
        else:
            # Use multi-tensor all-reduce for efficiency
            handle = dist.all_reduce_coalesced(bucket_grads, op=dist.ReduceOp.AVG, async_op=True)

        bucket_handles[bucket_idx] = handle

# Register hooks for all parameters
print0("Registering bucketed gradient hooks...")
for param in all_params:
    param.register_post_accumulate_grad_hook(_gradient_hook)

def wait_for_gradients():
    """Wait for all gradient reductions to complete and reset bucket state"""
    for handle in bucket_handles:
        if handle is not None:
            handle.wait()

    # Reset state for next iteration
    for i in range(len(bucket_ready_count)):
        bucket_ready_count[i] = 0
        bucket_handles[i] = None

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        ## Make sure all gradient reductions from the previous training step are finished before validation
        #wait_for_gradients()

        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(step)).backward()
    #for param in model.parameters():
    #    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    wait_for_gradients() # does the same thing as commented two lines above, but faster

    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()

====================================================================================================
Running Python 3.12.7 (main, May 24 2025, 20:59:58) [GCC 13.2.0]
Running PyTorch 2.8.0.dev20250524+cu126 compiled for CUDA 12.6
Sun May 25 21:52:09 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:61:00.0 Off |                    0 |
| N/A   34C    P0            145W /  700W |    5856MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  |   00000000:62:00.0 Off |                    0 |
| N/A   34C    P0            125W /  700W |    1518MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA H100 80GB HBM3          On  |   00000000:63:00.0 Off |                    0 |
| N/A   35C    P0            129W /  700W |    1518MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA H100 80GB HBM3          On  |   00000000:64:00.0 Off |                    0 |
| N/A   29C    P0            115W /  700W |    1518MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA H100 80GB HBM3          On  |   00000000:6A:00.0 Off |                    0 |
| N/A   30C    P0            125W /  700W |    1518MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA H100 80GB HBM3          On  |   00000000:6B:00.0 Off |                    0 |
| N/A   36C    P0            123W /  700W |    1518MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA H100 80GB HBM3          On  |   00000000:6C:00.0 Off |                    0 |
| N/A   35C    P0            125W /  700W |    1518MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA H100 80GB HBM3          On  |   00000000:6D:00.0 Off |                    0 |
| N/A   30C    P0            125W /  700W |    1518MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           40006      C   /usr/local/bin/python                  1508MiB |
|    0   N/A  N/A           40007      C   /usr/local/bin/python                   614MiB |
|    0   N/A  N/A           40008      C   /usr/local/bin/python                   614MiB |
|    0   N/A  N/A           40009      C   /usr/local/bin/python                   614MiB |
|    0   N/A  N/A           40010      C   /usr/local/bin/python                   614MiB |
|    0   N/A  N/A           40011      C   /usr/local/bin/python                   614MiB |
|    0   N/A  N/A           40012      C   /usr/local/bin/python                   614MiB |
|    0   N/A  N/A           40013      C   /usr/local/bin/python                   614MiB |
|    1   N/A  N/A           40007      C   /usr/local/bin/python                  1508MiB |
|    2   N/A  N/A           40008      C   /usr/local/bin/python                  1508MiB |
|    3   N/A  N/A           40009      C   /usr/local/bin/python                  1508MiB |
|    4   N/A  N/A           40010      C   /usr/local/bin/python                  1508MiB |
|    5   N/A  N/A           40011      C   /usr/local/bin/python                  1508MiB |
|    6   N/A  N/A           40012      C   /usr/local/bin/python                  1508MiB |
|    7   N/A  N/A           40013      C   /usr/local/bin/python                  1508MiB |
+-----------------------------------------------------------------------------------------+

====================================================================================================
Created 22 gradient buckets
Bucket 0: 1 params, 147.4 MB
Bucket 1: 1 params, 73.6 MB
Bucket 2: 1 params, 73.6 MB
Bucket 3: 1 params, 73.6 MB
Bucket 4: 1 params, 73.6 MB
Bucket 5: 2 params, 18.0 MB
Bucket 6: 2 params, 18.0 MB
Bucket 7: 2 params, 18.0 MB
Bucket 8: 2 params, 18.0 MB
Bucket 9: 2 params, 18.0 MB
Bucket 10: 2 params, 18.0 MB
Bucket 11: 2 params, 18.0 MB
Bucket 12: 2 params, 18.0 MB
Bucket 13: 2 params, 18.0 MB
Bucket 14: 2 params, 18.0 MB
Bucket 15: 2 params, 18.0 MB
Bucket 16: 3 params, 24.8 MB
Bucket 17: 3 params, 20.2 MB
Bucket 18: 3 params, 20.2 MB
Bucket 19: 3 params, 20.2 MB
Bucket 20: 9 params, 24.8 MB
Bucket 21: 27 params, 6.8 MB
Registering bucketed gradient hooks...
step:0/1770 val_loss:10.8258 train_time:0ms step_avg:0.03ms
step:1/1770 train_time:433ms step_avg:433.40ms
step:2/1770 train_time:501ms step_avg:250.75ms
step:3/1770 train_time:590ms step_avg:196.67ms
step:4/1770 train_time:683ms step_avg:170.78ms
step:5/1770 train_time:776ms step_avg:155.29ms
step:6/1770 train_time:871ms step_avg:145.17ms
step:7/1770 train_time:966ms step_avg:137.96ms
step:8/1770 train_time:1060ms step_avg:132.46ms
step:9/1770 train_time:1154ms step_avg:128.22ms
step:10/1770 train_time:1248ms step_avg:124.81ms
step:11/1770 train_time:1342ms step_avg:121.98ms
step:12/1770 train_time:1437ms step_avg:119.77ms
step:13/1770 train_time:1533ms step_avg:117.93ms
step:14/1770 train_time:1629ms step_avg:116.34ms
step:15/1770 train_time:1723ms step_avg:114.90ms
step:16/1770 train_time:1818ms step_avg:113.64ms
step:17/1770 train_time:1913ms step_avg:112.52ms
step:18/1770 train_time:2008ms step_avg:111.55ms
step:19/1770 train_time:2102ms step_avg:110.61ms
step:20/1770 train_time:2196ms step_avg:109.81ms
step:21/1770 train_time:2291ms step_avg:109.08ms
step:22/1770 train_time:2384ms step_avg:108.38ms
step:23/1770 train_time:2479ms step_avg:107.80ms
step:24/1770 train_time:2576ms step_avg:107.32ms
step:25/1770 train_time:2672ms step_avg:106.86ms
step:26/1770 train_time:2766ms step_avg:106.38ms
step:27/1770 train_time:2860ms step_avg:105.91ms
step:28/1770 train_time:2955ms step_avg:105.54ms
step:29/1770 train_time:3050ms step_avg:105.18ms
step:30/1770 train_time:3144ms step_avg:104.80ms
step:31/1770 train_time:3238ms step_avg:104.46ms
step:32/1770 train_time:3333ms step_avg:104.17ms
step:33/1770 train_time:3428ms step_avg:103.88ms
step:34/1770 train_time:3521ms step_avg:103.57ms
step:35/1770 train_time:3617ms step_avg:103.33ms
step:36/1770 train_time:3711ms step_avg:103.10ms
step:37/1770 train_time:3806ms step_avg:102.87ms
step:38/1770 train_time:3900ms step_avg:102.64ms
step:39/1770 train_time:3994ms step_avg:102.42ms
step:40/1770 train_time:4088ms step_avg:102.21ms
step:41/1770 train_time:4182ms step_avg:102.01ms
step:42/1770 train_time:4277ms step_avg:101.82ms
step:43/1770 train_time:4372ms step_avg:101.67ms
step:44/1770 train_time:4466ms step_avg:101.51ms
step:45/1770 train_time:4561ms step_avg:101.35ms
step:46/1770 train_time:4656ms step_avg:101.21ms
step:47/1770 train_time:4750ms step_avg:101.06ms
step:48/1770 train_time:4844ms step_avg:100.93ms
step:49/1770 train_time:4938ms step_avg:100.78ms
step:50/1770 train_time:5033ms step_avg:100.66ms
step:51/1770 train_time:5128ms step_avg:100.55ms
step:52/1770 train_time:5224ms step_avg:100.46ms
step:53/1770 train_time:5318ms step_avg:100.34ms
step:54/1770 train_time:5413ms step_avg:100.24ms
step:55/1770 train_time:5509ms step_avg:100.16ms
step:56/1770 train_time:5603ms step_avg:100.06ms
step:57/1770 train_time:5698ms step_avg:99.96ms
step:58/1770 train_time:5792ms step_avg:99.86ms
step:59/1770 train_time:5886ms step_avg:99.76ms
step:60/1770 train_time:5980ms step_avg:99.67ms
step:61/1770 train_time:6075ms step_avg:99.59ms
step:62/1770 train_time:6171ms step_avg:99.53ms
step:63/1770 train_time:6265ms step_avg:99.44ms
step:64/1770 train_time:6359ms step_avg:99.37ms
step:65/1770 train_time:6454ms step_avg:99.29ms
step:66/1770 train_time:6548ms step_avg:99.22ms
step:67/1770 train_time:6643ms step_avg:99.14ms
step:68/1770 train_time:6737ms step_avg:99.07ms
step:69/1770 train_time:6833ms step_avg:99.03ms
step:70/1770 train_time:6928ms step_avg:98.97ms
step:71/1770 train_time:7022ms step_avg:98.90ms
step:72/1770 train_time:7116ms step_avg:98.83ms
step:73/1770 train_time:7211ms step_avg:98.78ms
step:74/1770 train_time:7306ms step_avg:98.73ms
step:75/1770 train_time:7400ms step_avg:98.67ms
step:76/1770 train_time:7496ms step_avg:98.63ms
step:77/1770 train_time:7591ms step_avg:98.58ms
step:78/1770 train_time:7685ms step_avg:98.53ms
step:79/1770 train_time:7780ms step_avg:98.48ms
step:80/1770 train_time:7875ms step_avg:98.44ms
step:81/1770 train_time:7970ms step_avg:98.40ms
step:82/1770 train_time:8064ms step_avg:98.34ms
step:83/1770 train_time:8158ms step_avg:98.29ms
step:84/1770 train_time:8253ms step_avg:98.25ms
step:85/1770 train_time:8348ms step_avg:98.21ms
step:86/1770 train_time:8442ms step_avg:98.17ms
step:87/1770 train_time:8536ms step_avg:98.12ms
step:88/1770 train_time:8631ms step_avg:98.08ms
step:89/1770 train_time:8726ms step_avg:98.04ms
step:90/1770 train_time:8820ms step_avg:98.00ms
step:91/1770 train_time:8916ms step_avg:97.97ms
step:92/1770 train_time:9010ms step_avg:97.94ms
step:93/1770 train_time:9105ms step_avg:97.90ms
step:94/1770 train_time:9199ms step_avg:97.86ms
step:95/1770 train_time:9294ms step_avg:97.83ms
step:96/1770 train_time:9388ms step_avg:97.80ms
step:97/1770 train_time:9483ms step_avg:97.76ms
step:98/1770 train_time:9577ms step_avg:97.73ms
step:99/1770 train_time:9673ms step_avg:97.71ms
step:100/1770 train_time:9768ms step_avg:97.68ms
step:101/1770 train_time:9862ms step_avg:97.64ms
step:102/1770 train_time:9957ms step_avg:97.62ms
step:103/1770 train_time:10052ms step_avg:97.59ms
step:104/1770 train_time:10146ms step_avg:97.56ms
step:105/1770 train_time:10240ms step_avg:97.53ms
step:106/1770 train_time:10335ms step_avg:97.50ms
step:107/1770 train_time:10429ms step_avg:97.47ms
step:108/1770 train_time:10523ms step_avg:97.44ms
step:109/1770 train_time:10618ms step_avg:97.41ms
step:110/1770 train_time:10714ms step_avg:97.40ms
step:111/1770 train_time:10809ms step_avg:97.38ms
step:112/1770 train_time:10903ms step_avg:97.34ms
step:113/1770 train_time:10997ms step_avg:97.32ms
step:114/1770 train_time:11092ms step_avg:97.30ms
step:115/1770 train_time:11187ms step_avg:97.28ms
step:116/1770 train_time:11281ms step_avg:97.25ms
step:117/1770 train_time:11374ms step_avg:97.22ms
step:118/1770 train_time:11469ms step_avg:97.20ms
step:119/1770 train_time:11563ms step_avg:97.17ms
step:120/1770 train_time:11657ms step_avg:97.14ms
step:121/1770 train_time:11752ms step_avg:97.12ms
step:122/1770 train_time:11847ms step_avg:97.10ms
step:123/1770 train_time:11941ms step_avg:97.09ms
step:124/1770 train_time:12036ms step_avg:97.07ms
step:125/1770 train_time:12130ms step_avg:97.04ms
step:125/1770 val_loss:4.6555 train_time:12222ms step_avg:97.78ms
step:126/1770 train_time:12249ms step_avg:97.21ms
step:127/1770 train_time:12324ms step_avg:97.04ms
step:128/1770 train_time:12430ms step_avg:97.11ms
step:129/1770 train_time:12529ms step_avg:97.12ms
step:130/1770 train_time:12625ms step_avg:97.12ms
step:131/1770 train_time:12718ms step_avg:97.09ms
step:132/1770 train_time:12812ms step_avg:97.06ms
step:133/1770 train_time:12906ms step_avg:97.04ms
step:134/1770 train_time:13001ms step_avg:97.02ms
step:135/1770 train_time:13095ms step_avg:97.00ms
step:136/1770 train_time:13189ms step_avg:96.98ms
step:137/1770 train_time:13285ms step_avg:96.97ms
step:138/1770 train_time:13383ms step_avg:96.97ms
step:139/1770 train_time:13480ms step_avg:96.98ms
step:140/1770 train_time:13577ms step_avg:96.98ms
step:141/1770 train_time:13672ms step_avg:96.96ms
step:142/1770 train_time:13766ms step_avg:96.95ms
step:143/1770 train_time:13862ms step_avg:96.93ms
step:144/1770 train_time:13956ms step_avg:96.92ms
step:145/1770 train_time:14051ms step_avg:96.90ms
step:146/1770 train_time:14145ms step_avg:96.88ms
step:147/1770 train_time:14241ms step_avg:96.88ms
step:148/1770 train_time:14338ms step_avg:96.88ms
step:149/1770 train_time:14434ms step_avg:96.87ms
step:150/1770 train_time:14530ms step_avg:96.87ms
step:151/1770 train_time:14626ms step_avg:96.86ms
step:152/1770 train_time:14723ms step_avg:96.86ms
step:153/1770 train_time:14819ms step_avg:96.86ms
step:154/1770 train_time:14914ms step_avg:96.84ms
step:155/1770 train_time:15008ms step_avg:96.83ms
step:156/1770 train_time:15103ms step_avg:96.82ms
step:157/1770 train_time:15199ms step_avg:96.81ms
step:158/1770 train_time:15294ms step_avg:96.80ms
step:159/1770 train_time:15389ms step_avg:96.79ms
step:160/1770 train_time:15485ms step_avg:96.78ms
step:161/1770 train_time:15582ms step_avg:96.78ms
step:162/1770 train_time:15679ms step_avg:96.78ms
step:163/1770 train_time:15773ms step_avg:96.76ms
step:164/1770 train_time:15867ms step_avg:96.75ms
step:165/1770 train_time:15963ms step_avg:96.75ms
step:166/1770 train_time:16058ms step_avg:96.74ms
step:167/1770 train_time:16153ms step_avg:96.72ms
step:168/1770 train_time:16248ms step_avg:96.71ms
step:169/1770 train_time:16342ms step_avg:96.70ms
step:170/1770 train_time:16438ms step_avg:96.69ms
step:171/1770 train_time:16533ms step_avg:96.69ms
step:172/1770 train_time:16628ms step_avg:96.67ms
step:173/1770 train_time:16723ms step_avg:96.67ms
step:174/1770 train_time:16819ms step_avg:96.66ms
step:175/1770 train_time:16915ms step_avg:96.66ms
step:176/1770 train_time:17010ms step_avg:96.65ms
step:177/1770 train_time:17105ms step_avg:96.64ms
step:178/1770 train_time:17200ms step_avg:96.63ms
step:179/1770 train_time:17295ms step_avg:96.62ms
step:180/1770 train_time:17390ms step_avg:96.61ms
step:181/1770 train_time:17485ms step_avg:96.60ms
step:182/1770 train_time:17582ms step_avg:96.60ms
step:183/1770 train_time:17678ms step_avg:96.60ms
step:184/1770 train_time:17772ms step_avg:96.59ms
step:185/1770 train_time:17867ms step_avg:96.58ms
step:186/1770 train_time:17963ms step_avg:96.58ms
step:187/1770 train_time:18058ms step_avg:96.57ms
step:188/1770 train_time:18154ms step_avg:96.56ms
step:189/1770 train_time:18248ms step_avg:96.55ms
step:190/1770 train_time:18343ms step_avg:96.54ms
step:191/1770 train_time:18438ms step_avg:96.54ms
step:192/1770 train_time:18533ms step_avg:96.53ms
step:193/1770 train_time:18628ms step_avg:96.52ms
step:194/1770 train_time:18724ms step_avg:96.51ms
step:195/1770 train_time:18819ms step_avg:96.51ms
step:196/1770 train_time:18914ms step_avg:96.50ms
step:197/1770 train_time:19009ms step_avg:96.49ms
step:198/1770 train_time:19104ms step_avg:96.49ms
step:199/1770 train_time:19200ms step_avg:96.48ms
step:200/1770 train_time:19295ms step_avg:96.47ms
step:201/1770 train_time:19389ms step_avg:96.46ms
step:202/1770 train_time:19484ms step_avg:96.46ms
step:203/1770 train_time:19579ms step_avg:96.45ms
step:204/1770 train_time:19675ms step_avg:96.45ms
step:205/1770 train_time:19769ms step_avg:96.44ms
step:206/1770 train_time:19864ms step_avg:96.43ms
step:207/1770 train_time:19960ms step_avg:96.42ms
step:208/1770 train_time:20056ms step_avg:96.42ms
step:209/1770 train_time:20151ms step_avg:96.41ms
step:210/1770 train_time:20246ms step_avg:96.41ms
step:211/1770 train_time:20342ms step_avg:96.41ms
step:212/1770 train_time:20438ms step_avg:96.40ms
step:213/1770 train_time:20532ms step_avg:96.40ms
step:214/1770 train_time:20627ms step_avg:96.39ms
step:215/1770 train_time:20723ms step_avg:96.39ms
step:216/1770 train_time:20819ms step_avg:96.38ms
step:217/1770 train_time:20914ms step_avg:96.38ms
step:218/1770 train_time:21009ms step_avg:96.37ms
step:219/1770 train_time:21104ms step_avg:96.37ms
step:220/1770 train_time:21199ms step_avg:96.36ms
step:221/1770 train_time:21294ms step_avg:96.35ms
step:222/1770 train_time:21388ms step_avg:96.34ms
step:223/1770 train_time:21484ms step_avg:96.34ms
step:224/1770 train_time:21580ms step_avg:96.34ms
step:225/1770 train_time:21675ms step_avg:96.33ms
step:226/1770 train_time:21770ms step_avg:96.33ms
step:227/1770 train_time:21865ms step_avg:96.32ms
step:228/1770 train_time:21960ms step_avg:96.32ms
step:229/1770 train_time:22056ms step_avg:96.31ms
step:230/1770 train_time:22150ms step_avg:96.30ms
step:231/1770 train_time:22245ms step_avg:96.30ms
step:232/1770 train_time:22341ms step_avg:96.30ms
step:233/1770 train_time:22436ms step_avg:96.29ms
step:234/1770 train_time:22531ms step_avg:96.29ms
step:235/1770 train_time:22625ms step_avg:96.28ms
step:236/1770 train_time:22722ms step_avg:96.28ms
step:237/1770 train_time:22817ms step_avg:96.27ms
step:238/1770 train_time:22912ms step_avg:96.27ms
step:239/1770 train_time:23007ms step_avg:96.26ms
step:240/1770 train_time:23103ms step_avg:96.26ms
step:241/1770 train_time:23199ms step_avg:96.26ms
step:242/1770 train_time:23293ms step_avg:96.25ms
step:243/1770 train_time:23388ms step_avg:96.25ms
step:244/1770 train_time:23484ms step_avg:96.25ms
step:245/1770 train_time:23580ms step_avg:96.24ms
step:246/1770 train_time:23675ms step_avg:96.24ms
step:247/1770 train_time:23769ms step_avg:96.23ms
step:248/1770 train_time:23865ms step_avg:96.23ms
step:249/1770 train_time:23961ms step_avg:96.23ms
step:250/1770 train_time:24056ms step_avg:96.23ms
step:250/1770 val_loss:4.1062 train_time:24148ms step_avg:96.59ms
step:251/1770 train_time:24175ms step_avg:96.32ms
step:252/1770 train_time:24253ms step_avg:96.24ms
step:253/1770 train_time:24351ms step_avg:96.25ms
step:254/1770 train_time:24446ms step_avg:96.24ms
step:255/1770 train_time:24541ms step_avg:96.24ms
step:256/1770 train_time:24635ms step_avg:96.23ms
step:257/1770 train_time:24730ms step_avg:96.23ms
step:258/1770 train_time:24824ms step_avg:96.22ms
step:259/1770 train_time:24918ms step_avg:96.21ms
step:260/1770 train_time:25012ms step_avg:96.20ms
step:261/1770 train_time:25107ms step_avg:96.19ms
step:262/1770 train_time:25204ms step_avg:96.20ms
step:263/1770 train_time:25301ms step_avg:96.20ms
step:264/1770 train_time:25397ms step_avg:96.20ms
step:265/1770 train_time:25493ms step_avg:96.20ms
step:266/1770 train_time:25589ms step_avg:96.20ms
step:267/1770 train_time:25684ms step_avg:96.19ms
step:268/1770 train_time:25779ms step_avg:96.19ms
step:269/1770 train_time:25874ms step_avg:96.19ms
step:270/1770 train_time:25969ms step_avg:96.18ms
step:271/1770 train_time:26064ms step_avg:96.18ms
step:272/1770 train_time:26160ms step_avg:96.18ms
step:273/1770 train_time:26258ms step_avg:96.18ms
step:274/1770 train_time:26354ms step_avg:96.18ms
step:275/1770 train_time:26449ms step_avg:96.18ms
step:276/1770 train_time:26545ms step_avg:96.18ms
step:277/1770 train_time:26641ms step_avg:96.18ms
step:278/1770 train_time:26737ms step_avg:96.18ms
step:279/1770 train_time:26832ms step_avg:96.17ms
step:280/1770 train_time:26927ms step_avg:96.17ms
step:281/1770 train_time:27022ms step_avg:96.16ms
step:282/1770 train_time:27118ms step_avg:96.16ms
step:283/1770 train_time:27214ms step_avg:96.16ms
step:284/1770 train_time:27309ms step_avg:96.16ms
step:285/1770 train_time:27405ms step_avg:96.16ms
step:286/1770 train_time:27501ms step_avg:96.16ms
step:287/1770 train_time:27597ms step_avg:96.16ms
step:288/1770 train_time:27693ms step_avg:96.15ms
step:289/1770 train_time:27788ms step_avg:96.15ms
step:290/1770 train_time:27883ms step_avg:96.15ms
step:291/1770 train_time:27979ms step_avg:96.15ms
step:292/1770 train_time:28075ms step_avg:96.15ms
step:293/1770 train_time:28171ms step_avg:96.15ms
step:294/1770 train_time:28266ms step_avg:96.14ms
step:295/1770 train_time:28361ms step_avg:96.14ms
step:296/1770 train_time:28459ms step_avg:96.15ms
step:297/1770 train_time:28555ms step_avg:96.15ms
step:298/1770 train_time:28651ms step_avg:96.14ms
step:299/1770 train_time:28746ms step_avg:96.14ms
step:300/1770 train_time:28842ms step_avg:96.14ms
step:301/1770 train_time:28937ms step_avg:96.14ms
step:302/1770 train_time:29033ms step_avg:96.13ms
step:303/1770 train_time:29128ms step_avg:96.13ms
step:304/1770 train_time:29224ms step_avg:96.13ms
step:305/1770 train_time:29319ms step_avg:96.13ms
step:306/1770 train_time:29416ms step_avg:96.13ms
step:307/1770 train_time:29513ms step_avg:96.13ms
step:308/1770 train_time:29608ms step_avg:96.13ms
step:309/1770 train_time:29704ms step_avg:96.13ms
step:310/1770 train_time:29799ms step_avg:96.13ms
step:311/1770 train_time:29895ms step_avg:96.13ms
step:312/1770 train_time:29991ms step_avg:96.12ms
step:313/1770 train_time:30086ms step_avg:96.12ms
step:314/1770 train_time:30181ms step_avg:96.12ms
step:315/1770 train_time:30278ms step_avg:96.12ms
step:316/1770 train_time:30373ms step_avg:96.12ms
step:317/1770 train_time:30469ms step_avg:96.12ms
step:318/1770 train_time:30566ms step_avg:96.12ms
step:319/1770 train_time:30662ms step_avg:96.12ms
step:320/1770 train_time:30759ms step_avg:96.12ms
step:321/1770 train_time:30855ms step_avg:96.12ms
step:322/1770 train_time:30950ms step_avg:96.12ms
step:323/1770 train_time:31045ms step_avg:96.11ms
step:324/1770 train_time:31140ms step_avg:96.11ms
step:325/1770 train_time:31236ms step_avg:96.11ms
step:326/1770 train_time:31332ms step_avg:96.11ms
step:327/1770 train_time:31427ms step_avg:96.11ms
step:328/1770 train_time:31523ms step_avg:96.11ms
step:329/1770 train_time:31618ms step_avg:96.10ms
step:330/1770 train_time:31715ms step_avg:96.10ms
step:331/1770 train_time:31810ms step_avg:96.10ms
step:332/1770 train_time:31905ms step_avg:96.10ms
step:333/1770 train_time:32001ms step_avg:96.10ms
step:334/1770 train_time:32097ms step_avg:96.10ms
step:335/1770 train_time:32193ms step_avg:96.10ms
step:336/1770 train_time:32288ms step_avg:96.10ms
step:337/1770 train_time:32384ms step_avg:96.09ms
step:338/1770 train_time:32480ms step_avg:96.09ms
step:339/1770 train_time:32576ms step_avg:96.09ms
step:340/1770 train_time:32671ms step_avg:96.09ms
step:341/1770 train_time:32766ms step_avg:96.09ms
step:342/1770 train_time:32862ms step_avg:96.09ms
step:343/1770 train_time:32959ms step_avg:96.09ms
step:344/1770 train_time:33056ms step_avg:96.09ms
step:345/1770 train_time:33152ms step_avg:96.09ms
step:346/1770 train_time:33247ms step_avg:96.09ms
step:347/1770 train_time:33343ms step_avg:96.09ms
step:348/1770 train_time:33439ms step_avg:96.09ms
step:349/1770 train_time:33537ms step_avg:96.09ms
step:350/1770 train_time:33633ms step_avg:96.09ms
step:351/1770 train_time:33728ms step_avg:96.09ms
step:352/1770 train_time:33824ms step_avg:96.09ms
step:353/1770 train_time:33920ms step_avg:96.09ms
step:354/1770 train_time:34016ms step_avg:96.09ms
step:355/1770 train_time:34112ms step_avg:96.09ms
step:356/1770 train_time:34207ms step_avg:96.09ms
step:357/1770 train_time:34302ms step_avg:96.08ms
step:358/1770 train_time:34397ms step_avg:96.08ms
step:359/1770 train_time:34493ms step_avg:96.08ms
step:360/1770 train_time:34589ms step_avg:96.08ms
step:361/1770 train_time:34684ms step_avg:96.08ms
step:362/1770 train_time:34781ms step_avg:96.08ms
step:363/1770 train_time:34877ms step_avg:96.08ms
step:364/1770 train_time:34973ms step_avg:96.08ms
step:365/1770 train_time:35069ms step_avg:96.08ms
step:366/1770 train_time:35164ms step_avg:96.08ms
step:367/1770 train_time:35261ms step_avg:96.08ms
step:368/1770 train_time:35357ms step_avg:96.08ms
step:369/1770 train_time:35453ms step_avg:96.08ms
step:370/1770 train_time:35548ms step_avg:96.08ms
step:371/1770 train_time:35643ms step_avg:96.07ms
step:372/1770 train_time:35740ms step_avg:96.07ms
step:373/1770 train_time:35836ms step_avg:96.07ms
step:374/1770 train_time:35932ms step_avg:96.07ms
step:375/1770 train_time:36028ms step_avg:96.07ms
step:375/1770 val_loss:3.9040 train_time:36120ms step_avg:96.32ms
step:376/1770 train_time:36147ms step_avg:96.14ms
step:377/1770 train_time:36226ms step_avg:96.09ms
step:378/1770 train_time:36325ms step_avg:96.10ms
step:379/1770 train_time:36421ms step_avg:96.10ms
step:380/1770 train_time:36517ms step_avg:96.10ms
step:381/1770 train_time:36612ms step_avg:96.10ms
step:382/1770 train_time:36707ms step_avg:96.09ms
step:383/1770 train_time:36802ms step_avg:96.09ms
step:384/1770 train_time:36897ms step_avg:96.09ms
step:385/1770 train_time:36992ms step_avg:96.08ms
step:386/1770 train_time:37087ms step_avg:96.08ms
step:387/1770 train_time:37184ms step_avg:96.08ms
step:388/1770 train_time:37283ms step_avg:96.09ms
step:389/1770 train_time:37379ms step_avg:96.09ms
step:390/1770 train_time:37476ms step_avg:96.09ms
step:391/1770 train_time:37572ms step_avg:96.09ms
step:392/1770 train_time:37667ms step_avg:96.09ms
step:393/1770 train_time:37762ms step_avg:96.09ms
step:394/1770 train_time:37857ms step_avg:96.08ms
step:395/1770 train_time:37952ms step_avg:96.08ms
step:396/1770 train_time:38051ms step_avg:96.09ms
step:397/1770 train_time:38148ms step_avg:96.09ms
step:398/1770 train_time:38247ms step_avg:96.10ms
step:399/1770 train_time:38346ms step_avg:96.10ms
step:400/1770 train_time:38444ms step_avg:96.11ms
step:401/1770 train_time:38542ms step_avg:96.11ms
step:402/1770 train_time:38639ms step_avg:96.12ms
step:403/1770 train_time:38736ms step_avg:96.12ms
step:404/1770 train_time:38833ms step_avg:96.12ms
step:405/1770 train_time:38931ms step_avg:96.13ms
step:406/1770 train_time:39029ms step_avg:96.13ms
step:407/1770 train_time:39127ms step_avg:96.13ms
step:408/1770 train_time:39224ms step_avg:96.14ms
step:409/1770 train_time:39322ms step_avg:96.14ms
step:410/1770 train_time:39420ms step_avg:96.15ms
step:411/1770 train_time:39518ms step_avg:96.15ms
step:412/1770 train_time:39616ms step_avg:96.16ms
step:413/1770 train_time:39714ms step_avg:96.16ms
step:414/1770 train_time:39812ms step_avg:96.16ms
step:415/1770 train_time:39910ms step_avg:96.17ms
step:416/1770 train_time:40007ms step_avg:96.17ms
step:417/1770 train_time:40105ms step_avg:96.18ms
step:418/1770 train_time:40203ms step_avg:96.18ms
step:419/1770 train_time:40301ms step_avg:96.18ms
step:420/1770 train_time:40399ms step_avg:96.19ms
step:421/1770 train_time:40497ms step_avg:96.19ms
step:422/1770 train_time:40597ms step_avg:96.20ms
step:423/1770 train_time:40695ms step_avg:96.21ms
step:424/1770 train_time:40794ms step_avg:96.21ms
step:425/1770 train_time:40892ms step_avg:96.22ms
step:426/1770 train_time:40990ms step_avg:96.22ms
step:427/1770 train_time:41088ms step_avg:96.23ms
step:428/1770 train_time:41187ms step_avg:96.23ms
step:429/1770 train_time:41286ms step_avg:96.24ms
step:430/1770 train_time:41384ms step_avg:96.24ms
step:431/1770 train_time:41481ms step_avg:96.24ms
step:432/1770 train_time:41580ms step_avg:96.25ms
step:433/1770 train_time:41678ms step_avg:96.25ms
step:434/1770 train_time:41776ms step_avg:96.26ms
step:435/1770 train_time:41875ms step_avg:96.26ms
step:436/1770 train_time:41972ms step_avg:96.27ms
step:437/1770 train_time:42070ms step_avg:96.27ms
step:438/1770 train_time:42169ms step_avg:96.28ms
step:439/1770 train_time:42268ms step_avg:96.28ms
step:440/1770 train_time:42366ms step_avg:96.29ms
step:441/1770 train_time:42465ms step_avg:96.29ms
step:442/1770 train_time:42563ms step_avg:96.30ms
step:443/1770 train_time:42661ms step_avg:96.30ms
step:444/1770 train_time:42758ms step_avg:96.30ms
step:445/1770 train_time:42856ms step_avg:96.31ms
step:446/1770 train_time:42954ms step_avg:96.31ms
step:447/1770 train_time:43052ms step_avg:96.31ms
step:448/1770 train_time:43151ms step_avg:96.32ms
step:449/1770 train_time:43250ms step_avg:96.32ms
step:450/1770 train_time:43349ms step_avg:96.33ms
step:451/1770 train_time:43448ms step_avg:96.34ms
step:452/1770 train_time:43546ms step_avg:96.34ms
step:453/1770 train_time:43644ms step_avg:96.34ms
step:454/1770 train_time:43741ms step_avg:96.35ms
step:455/1770 train_time:43839ms step_avg:96.35ms
step:456/1770 train_time:43937ms step_avg:96.35ms
step:457/1770 train_time:44035ms step_avg:96.36ms
step:458/1770 train_time:44133ms step_avg:96.36ms
step:459/1770 train_time:44231ms step_avg:96.36ms
step:460/1770 train_time:44329ms step_avg:96.37ms
step:461/1770 train_time:44428ms step_avg:96.37ms
step:462/1770 train_time:44527ms step_avg:96.38ms
step:463/1770 train_time:44625ms step_avg:96.38ms
step:464/1770 train_time:44723ms step_avg:96.38ms
step:465/1770 train_time:44820ms step_avg:96.39ms
step:466/1770 train_time:44917ms step_avg:96.39ms
step:467/1770 train_time:45015ms step_avg:96.39ms
step:468/1770 train_time:45113ms step_avg:96.40ms
step:469/1770 train_time:45212ms step_avg:96.40ms
step:470/1770 train_time:45309ms step_avg:96.40ms
step:471/1770 train_time:45407ms step_avg:96.40ms
step:472/1770 train_time:45505ms step_avg:96.41ms
step:473/1770 train_time:45603ms step_avg:96.41ms
step:474/1770 train_time:45701ms step_avg:96.41ms
step:475/1770 train_time:45799ms step_avg:96.42ms
step:476/1770 train_time:45896ms step_avg:96.42ms
step:477/1770 train_time:45994ms step_avg:96.42ms
step:478/1770 train_time:46091ms step_avg:96.43ms
step:479/1770 train_time:46189ms step_avg:96.43ms
step:480/1770 train_time:46287ms step_avg:96.43ms
step:481/1770 train_time:46388ms step_avg:96.44ms
step:482/1770 train_time:46485ms step_avg:96.44ms
step:483/1770 train_time:46582ms step_avg:96.44ms
step:484/1770 train_time:46680ms step_avg:96.45ms
step:485/1770 train_time:46779ms step_avg:96.45ms
step:486/1770 train_time:46877ms step_avg:96.46ms
step:487/1770 train_time:46975ms step_avg:96.46ms
step:488/1770 train_time:47074ms step_avg:96.46ms
step:489/1770 train_time:47172ms step_avg:96.47ms
step:490/1770 train_time:47270ms step_avg:96.47ms
step:491/1770 train_time:47368ms step_avg:96.47ms
step:492/1770 train_time:47466ms step_avg:96.48ms
step:493/1770 train_time:47564ms step_avg:96.48ms
step:494/1770 train_time:47662ms step_avg:96.48ms
step:495/1770 train_time:47760ms step_avg:96.48ms
step:496/1770 train_time:47858ms step_avg:96.49ms
step:497/1770 train_time:47955ms step_avg:96.49ms
step:498/1770 train_time:48053ms step_avg:96.49ms
step:499/1770 train_time:48151ms step_avg:96.49ms
step:500/1770 train_time:48248ms step_avg:96.50ms
step:500/1770 val_loss:3.7512 train_time:48343ms step_avg:96.69ms
step:501/1770 train_time:48370ms step_avg:96.55ms
step:502/1770 train_time:48454ms step_avg:96.52ms
step:503/1770 train_time:48554ms step_avg:96.53ms
step:504/1770 train_time:48652ms step_avg:96.53ms
step:505/1770 train_time:48750ms step_avg:96.53ms
step:506/1770 train_time:48849ms step_avg:96.54ms
step:507/1770 train_time:48947ms step_avg:96.54ms
step:508/1770 train_time:49044ms step_avg:96.54ms
step:509/1770 train_time:49142ms step_avg:96.55ms
step:510/1770 train_time:49238ms step_avg:96.54ms
step:511/1770 train_time:49336ms step_avg:96.55ms
step:512/1770 train_time:49435ms step_avg:96.55ms
step:513/1770 train_time:49536ms step_avg:96.56ms
step:514/1770 train_time:49634ms step_avg:96.56ms
step:515/1770 train_time:49731ms step_avg:96.57ms
step:516/1770 train_time:49829ms step_avg:96.57ms
step:517/1770 train_time:49927ms step_avg:96.57ms
step:518/1770 train_time:50025ms step_avg:96.57ms
step:519/1770 train_time:50122ms step_avg:96.57ms
step:520/1770 train_time:50219ms step_avg:96.58ms
step:521/1770 train_time:50317ms step_avg:96.58ms
step:522/1770 train_time:50416ms step_avg:96.58ms
step:523/1770 train_time:50515ms step_avg:96.59ms
step:524/1770 train_time:50614ms step_avg:96.59ms
step:525/1770 train_time:50712ms step_avg:96.59ms
step:526/1770 train_time:50810ms step_avg:96.60ms
step:527/1770 train_time:50909ms step_avg:96.60ms
step:528/1770 train_time:51007ms step_avg:96.60ms
step:529/1770 train_time:51106ms step_avg:96.61ms
step:530/1770 train_time:51205ms step_avg:96.61ms
step:531/1770 train_time:51303ms step_avg:96.62ms
step:532/1770 train_time:51403ms step_avg:96.62ms
step:533/1770 train_time:51502ms step_avg:96.63ms
step:534/1770 train_time:51601ms step_avg:96.63ms
step:535/1770 train_time:51700ms step_avg:96.64ms
step:536/1770 train_time:51798ms step_avg:96.64ms
step:537/1770 train_time:51896ms step_avg:96.64ms
step:538/1770 train_time:51994ms step_avg:96.64ms
step:539/1770 train_time:52092ms step_avg:96.64ms
step:540/1770 train_time:52189ms step_avg:96.65ms
step:541/1770 train_time:52288ms step_avg:96.65ms
step:542/1770 train_time:52387ms step_avg:96.65ms
step:543/1770 train_time:52486ms step_avg:96.66ms
step:544/1770 train_time:52586ms step_avg:96.67ms
step:545/1770 train_time:52686ms step_avg:96.67ms
step:546/1770 train_time:52786ms step_avg:96.68ms
step:547/1770 train_time:52885ms step_avg:96.68ms
step:548/1770 train_time:52984ms step_avg:96.69ms
step:549/1770 train_time:53083ms step_avg:96.69ms
step:550/1770 train_time:53180ms step_avg:96.69ms
step:551/1770 train_time:53279ms step_avg:96.69ms
step:552/1770 train_time:53378ms step_avg:96.70ms
step:553/1770 train_time:53476ms step_avg:96.70ms
step:554/1770 train_time:53574ms step_avg:96.70ms
step:555/1770 train_time:53672ms step_avg:96.71ms
step:556/1770 train_time:53771ms step_avg:96.71ms
step:557/1770 train_time:53869ms step_avg:96.71ms
step:558/1770 train_time:53969ms step_avg:96.72ms
step:559/1770 train_time:54068ms step_avg:96.72ms
step:560/1770 train_time:54167ms step_avg:96.73ms
step:561/1770 train_time:54265ms step_avg:96.73ms
step:562/1770 train_time:54365ms step_avg:96.73ms
step:563/1770 train_time:54464ms step_avg:96.74ms
step:564/1770 train_time:54563ms step_avg:96.74ms
step:565/1770 train_time:54662ms step_avg:96.75ms
step:566/1770 train_time:54761ms step_avg:96.75ms
step:567/1770 train_time:54859ms step_avg:96.75ms
step:568/1770 train_time:54958ms step_avg:96.76ms
step:569/1770 train_time:55056ms step_avg:96.76ms
step:570/1770 train_time:55154ms step_avg:96.76ms
step:571/1770 train_time:55252ms step_avg:96.76ms
step:572/1770 train_time:55350ms step_avg:96.77ms
step:573/1770 train_time:55449ms step_avg:96.77ms
step:574/1770 train_time:55548ms step_avg:96.77ms
step:575/1770 train_time:55648ms step_avg:96.78ms
step:576/1770 train_time:55747ms step_avg:96.78ms
step:577/1770 train_time:55846ms step_avg:96.79ms
step:578/1770 train_time:55946ms step_avg:96.79ms
step:579/1770 train_time:56045ms step_avg:96.80ms
step:580/1770 train_time:56143ms step_avg:96.80ms
step:581/1770 train_time:56241ms step_avg:96.80ms
step:582/1770 train_time:56339ms step_avg:96.80ms
step:583/1770 train_time:56437ms step_avg:96.80ms
step:584/1770 train_time:56535ms step_avg:96.81ms
step:585/1770 train_time:56633ms step_avg:96.81ms
step:586/1770 train_time:56731ms step_avg:96.81ms
step:587/1770 train_time:56830ms step_avg:96.81ms
step:588/1770 train_time:56929ms step_avg:96.82ms
step:589/1770 train_time:57030ms step_avg:96.82ms
step:590/1770 train_time:57128ms step_avg:96.83ms
step:591/1770 train_time:57227ms step_avg:96.83ms
step:592/1770 train_time:57327ms step_avg:96.84ms
step:593/1770 train_time:57426ms step_avg:96.84ms
step:594/1770 train_time:57525ms step_avg:96.84ms
step:595/1770 train_time:57626ms step_avg:96.85ms
step:596/1770 train_time:57726ms step_avg:96.86ms
step:597/1770 train_time:57825ms step_avg:96.86ms
step:598/1770 train_time:57924ms step_avg:96.86ms
step:599/1770 train_time:58023ms step_avg:96.87ms
step:600/1770 train_time:58120ms step_avg:96.87ms
step:601/1770 train_time:58218ms step_avg:96.87ms
step:602/1770 train_time:58316ms step_avg:96.87ms
step:603/1770 train_time:58414ms step_avg:96.87ms
step:604/1770 train_time:58512ms step_avg:96.87ms
step:605/1770 train_time:58611ms step_avg:96.88ms
step:606/1770 train_time:58710ms step_avg:96.88ms
step:607/1770 train_time:58809ms step_avg:96.88ms
step:608/1770 train_time:58908ms step_avg:96.89ms
step:609/1770 train_time:59007ms step_avg:96.89ms
step:610/1770 train_time:59106ms step_avg:96.90ms
step:611/1770 train_time:59206ms step_avg:96.90ms
step:612/1770 train_time:59305ms step_avg:96.90ms
step:613/1770 train_time:59405ms step_avg:96.91ms
step:614/1770 train_time:59504ms step_avg:96.91ms
step:615/1770 train_time:59603ms step_avg:96.92ms
step:616/1770 train_time:59702ms step_avg:96.92ms
step:617/1770 train_time:59800ms step_avg:96.92ms
step:618/1770 train_time:59898ms step_avg:96.92ms
step:619/1770 train_time:59996ms step_avg:96.92ms
step:620/1770 train_time:60095ms step_avg:96.93ms
step:621/1770 train_time:60192ms step_avg:96.93ms
step:622/1770 train_time:60290ms step_avg:96.93ms
step:623/1770 train_time:60389ms step_avg:96.93ms
step:624/1770 train_time:60489ms step_avg:96.94ms
step:625/1770 train_time:60588ms step_avg:96.94ms
step:625/1770 val_loss:3.6636 train_time:60684ms step_avg:97.09ms
step:626/1770 train_time:60710ms step_avg:96.98ms
step:627/1770 train_time:60792ms step_avg:96.96ms
step:628/1770 train_time:60892ms step_avg:96.96ms
step:629/1770 train_time:60990ms step_avg:96.96ms
step:630/1770 train_time:61088ms step_avg:96.97ms
step:631/1770 train_time:61186ms step_avg:96.97ms
step:632/1770 train_time:61283ms step_avg:96.97ms
step:633/1770 train_time:61380ms step_avg:96.97ms
step:634/1770 train_time:61478ms step_avg:96.97ms
step:635/1770 train_time:61576ms step_avg:96.97ms
step:636/1770 train_time:61676ms step_avg:96.97ms
step:637/1770 train_time:61777ms step_avg:96.98ms
step:638/1770 train_time:61877ms step_avg:96.99ms
step:639/1770 train_time:61977ms step_avg:96.99ms
step:640/1770 train_time:62076ms step_avg:96.99ms
step:641/1770 train_time:62176ms step_avg:97.00ms
step:642/1770 train_time:62275ms step_avg:97.00ms
step:643/1770 train_time:62374ms step_avg:97.00ms
step:644/1770 train_time:62473ms step_avg:97.01ms
step:645/1770 train_time:62571ms step_avg:97.01ms
step:646/1770 train_time:62670ms step_avg:97.01ms
step:647/1770 train_time:62770ms step_avg:97.02ms
step:648/1770 train_time:62869ms step_avg:97.02ms
step:649/1770 train_time:62968ms step_avg:97.02ms
step:650/1770 train_time:63067ms step_avg:97.03ms
step:651/1770 train_time:63166ms step_avg:97.03ms
step:652/1770 train_time:63264ms step_avg:97.03ms
step:653/1770 train_time:63362ms step_avg:97.03ms
step:654/1770 train_time:63460ms step_avg:97.03ms
step:655/1770 train_time:63558ms step_avg:97.03ms
step:656/1770 train_time:63657ms step_avg:97.04ms
step:657/1770 train_time:63756ms step_avg:97.04ms
step:658/1770 train_time:63858ms step_avg:97.05ms
step:659/1770 train_time:63960ms step_avg:97.06ms
step:660/1770 train_time:64062ms step_avg:97.06ms
step:661/1770 train_time:64163ms step_avg:97.07ms
step:662/1770 train_time:64263ms step_avg:97.07ms
step:663/1770 train_time:64363ms step_avg:97.08ms
step:664/1770 train_time:64463ms step_avg:97.08ms
step:665/1770 train_time:64563ms step_avg:97.09ms
step:666/1770 train_time:64662ms step_avg:97.09ms
step:667/1770 train_time:64763ms step_avg:97.10ms
step:668/1770 train_time:64863ms step_avg:97.10ms
step:669/1770 train_time:64964ms step_avg:97.11ms
step:670/1770 train_time:65064ms step_avg:97.11ms
step:671/1770 train_time:65164ms step_avg:97.12ms
step:672/1770 train_time:65264ms step_avg:97.12ms
step:673/1770 train_time:65363ms step_avg:97.12ms
step:674/1770 train_time:65463ms step_avg:97.13ms
step:675/1770 train_time:65562ms step_avg:97.13ms
step:676/1770 train_time:65662ms step_avg:97.13ms
step:677/1770 train_time:65762ms step_avg:97.14ms
step:678/1770 train_time:65862ms step_avg:97.14ms
step:679/1770 train_time:65962ms step_avg:97.15ms
step:680/1770 train_time:66064ms step_avg:97.15ms
step:681/1770 train_time:66163ms step_avg:97.16ms
step:682/1770 train_time:66263ms step_avg:97.16ms
step:683/1770 train_time:66364ms step_avg:97.16ms
step:684/1770 train_time:66464ms step_avg:97.17ms
step:685/1770 train_time:66563ms step_avg:97.17ms
step:686/1770 train_time:66664ms step_avg:97.18ms
step:687/1770 train_time:66764ms step_avg:97.18ms
step:688/1770 train_time:66864ms step_avg:97.19ms
step:689/1770 train_time:66964ms step_avg:97.19ms
step:690/1770 train_time:67065ms step_avg:97.20ms
step:691/1770 train_time:67165ms step_avg:97.20ms
step:692/1770 train_time:67265ms step_avg:97.20ms
step:693/1770 train_time:67365ms step_avg:97.21ms
step:694/1770 train_time:67465ms step_avg:97.21ms
step:695/1770 train_time:67564ms step_avg:97.21ms
step:696/1770 train_time:67664ms step_avg:97.22ms
step:697/1770 train_time:67763ms step_avg:97.22ms
step:698/1770 train_time:67863ms step_avg:97.22ms
step:699/1770 train_time:67963ms step_avg:97.23ms
step:700/1770 train_time:68063ms step_avg:97.23ms
step:701/1770 train_time:68163ms step_avg:97.24ms
step:702/1770 train_time:68264ms step_avg:97.24ms
step:703/1770 train_time:68364ms step_avg:97.25ms
step:704/1770 train_time:68464ms step_avg:97.25ms
step:705/1770 train_time:68564ms step_avg:97.25ms
step:706/1770 train_time:68664ms step_avg:97.26ms
step:707/1770 train_time:68763ms step_avg:97.26ms
step:708/1770 train_time:68863ms step_avg:97.26ms
step:709/1770 train_time:68962ms step_avg:97.27ms
step:710/1770 train_time:69063ms step_avg:97.27ms
step:711/1770 train_time:69163ms step_avg:97.28ms
step:712/1770 train_time:69263ms step_avg:97.28ms
step:713/1770 train_time:69363ms step_avg:97.28ms
step:714/1770 train_time:69463ms step_avg:97.29ms
step:715/1770 train_time:69563ms step_avg:97.29ms
step:716/1770 train_time:69663ms step_avg:97.30ms
step:717/1770 train_time:69763ms step_avg:97.30ms
step:718/1770 train_time:69863ms step_avg:97.30ms
step:719/1770 train_time:69963ms step_avg:97.31ms
step:720/1770 train_time:70063ms step_avg:97.31ms
step:721/1770 train_time:70164ms step_avg:97.31ms
step:722/1770 train_time:70264ms step_avg:97.32ms
step:723/1770 train_time:70364ms step_avg:97.32ms
step:724/1770 train_time:70464ms step_avg:97.33ms
step:725/1770 train_time:70563ms step_avg:97.33ms
step:726/1770 train_time:70664ms step_avg:97.33ms
step:727/1770 train_time:70764ms step_avg:97.34ms
step:728/1770 train_time:70864ms step_avg:97.34ms
step:729/1770 train_time:70964ms step_avg:97.34ms
step:730/1770 train_time:71064ms step_avg:97.35ms
step:731/1770 train_time:71163ms step_avg:97.35ms
step:732/1770 train_time:71264ms step_avg:97.35ms
step:733/1770 train_time:71364ms step_avg:97.36ms
step:734/1770 train_time:71463ms step_avg:97.36ms
step:735/1770 train_time:71563ms step_avg:97.37ms
step:736/1770 train_time:71663ms step_avg:97.37ms
step:737/1770 train_time:71763ms step_avg:97.37ms
step:738/1770 train_time:71863ms step_avg:97.38ms
step:739/1770 train_time:71964ms step_avg:97.38ms
step:740/1770 train_time:72064ms step_avg:97.38ms
step:741/1770 train_time:72163ms step_avg:97.39ms
step:742/1770 train_time:72263ms step_avg:97.39ms
step:743/1770 train_time:72364ms step_avg:97.39ms
step:744/1770 train_time:72463ms step_avg:97.40ms
step:745/1770 train_time:72564ms step_avg:97.40ms
step:746/1770 train_time:72664ms step_avg:97.41ms
step:747/1770 train_time:72764ms step_avg:97.41ms
step:748/1770 train_time:72864ms step_avg:97.41ms
step:749/1770 train_time:72964ms step_avg:97.42ms
step:750/1770 train_time:73064ms step_avg:97.42ms
step:750/1770 val_loss:3.5992 train_time:73161ms step_avg:97.55ms
step:751/1770 train_time:73188ms step_avg:97.45ms
step:752/1770 train_time:73276ms step_avg:97.44ms
step:753/1770 train_time:73378ms step_avg:97.45ms
step:754/1770 train_time:73477ms step_avg:97.45ms
step:755/1770 train_time:73577ms step_avg:97.45ms
step:756/1770 train_time:73676ms step_avg:97.46ms
step:757/1770 train_time:73776ms step_avg:97.46ms
step:758/1770 train_time:73876ms step_avg:97.46ms
step:759/1770 train_time:73976ms step_avg:97.46ms
step:760/1770 train_time:74075ms step_avg:97.47ms
step:761/1770 train_time:74177ms step_avg:97.47ms
step:762/1770 train_time:74280ms step_avg:97.48ms
step:763/1770 train_time:74381ms step_avg:97.48ms
step:764/1770 train_time:74482ms step_avg:97.49ms
step:765/1770 train_time:74582ms step_avg:97.49ms
step:766/1770 train_time:74681ms step_avg:97.50ms
step:767/1770 train_time:74781ms step_avg:97.50ms
step:768/1770 train_time:74880ms step_avg:97.50ms
step:769/1770 train_time:74980ms step_avg:97.50ms
step:770/1770 train_time:75080ms step_avg:97.51ms
step:771/1770 train_time:75179ms step_avg:97.51ms
step:772/1770 train_time:75279ms step_avg:97.51ms
step:773/1770 train_time:75379ms step_avg:97.52ms
step:774/1770 train_time:75479ms step_avg:97.52ms
step:775/1770 train_time:75579ms step_avg:97.52ms
step:776/1770 train_time:75678ms step_avg:97.52ms
step:777/1770 train_time:75779ms step_avg:97.53ms
step:778/1770 train_time:75878ms step_avg:97.53ms
step:779/1770 train_time:75978ms step_avg:97.53ms
step:780/1770 train_time:76078ms step_avg:97.54ms
step:781/1770 train_time:76178ms step_avg:97.54ms
step:782/1770 train_time:76278ms step_avg:97.54ms
step:783/1770 train_time:76378ms step_avg:97.55ms
step:784/1770 train_time:76478ms step_avg:97.55ms
step:785/1770 train_time:76578ms step_avg:97.55ms
step:786/1770 train_time:76678ms step_avg:97.55ms
step:787/1770 train_time:76778ms step_avg:97.56ms
step:788/1770 train_time:76878ms step_avg:97.56ms
step:789/1770 train_time:76978ms step_avg:97.56ms
step:790/1770 train_time:77077ms step_avg:97.57ms
step:791/1770 train_time:77177ms step_avg:97.57ms
step:792/1770 train_time:77278ms step_avg:97.57ms
step:793/1770 train_time:77378ms step_avg:97.58ms
step:794/1770 train_time:77479ms step_avg:97.58ms
step:795/1770 train_time:77580ms step_avg:97.58ms
step:796/1770 train_time:77680ms step_avg:97.59ms
step:797/1770 train_time:77780ms step_avg:97.59ms
step:798/1770 train_time:77880ms step_avg:97.59ms
step:799/1770 train_time:77980ms step_avg:97.60ms
step:800/1770 train_time:78080ms step_avg:97.60ms
step:801/1770 train_time:78180ms step_avg:97.60ms
step:802/1770 train_time:78280ms step_avg:97.61ms
step:803/1770 train_time:78380ms step_avg:97.61ms
step:804/1770 train_time:78480ms step_avg:97.61ms
step:805/1770 train_time:78580ms step_avg:97.61ms
step:806/1770 train_time:78680ms step_avg:97.62ms
step:807/1770 train_time:78780ms step_avg:97.62ms
step:808/1770 train_time:78880ms step_avg:97.62ms
step:809/1770 train_time:78980ms step_avg:97.63ms
step:810/1770 train_time:79080ms step_avg:97.63ms
step:811/1770 train_time:79180ms step_avg:97.63ms
step:812/1770 train_time:79280ms step_avg:97.64ms
step:813/1770 train_time:79380ms step_avg:97.64ms
step:814/1770 train_time:79480ms step_avg:97.64ms
step:815/1770 train_time:79580ms step_avg:97.64ms
step:816/1770 train_time:79679ms step_avg:97.65ms
step:817/1770 train_time:79779ms step_avg:97.65ms
step:818/1770 train_time:79879ms step_avg:97.65ms
step:819/1770 train_time:79979ms step_avg:97.65ms
step:820/1770 train_time:80079ms step_avg:97.66ms
step:821/1770 train_time:80179ms step_avg:97.66ms
step:822/1770 train_time:80279ms step_avg:97.66ms
step:823/1770 train_time:80380ms step_avg:97.67ms
step:824/1770 train_time:80479ms step_avg:97.67ms
step:825/1770 train_time:80580ms step_avg:97.67ms
step:826/1770 train_time:80680ms step_avg:97.67ms
step:827/1770 train_time:80780ms step_avg:97.68ms
step:828/1770 train_time:80879ms step_avg:97.68ms
step:829/1770 train_time:80979ms step_avg:97.68ms
step:830/1770 train_time:81079ms step_avg:97.69ms
step:831/1770 train_time:81179ms step_avg:97.69ms
step:832/1770 train_time:81279ms step_avg:97.69ms
step:833/1770 train_time:81379ms step_avg:97.69ms
step:834/1770 train_time:81479ms step_avg:97.70ms
step:835/1770 train_time:81579ms step_avg:97.70ms
step:836/1770 train_time:81680ms step_avg:97.70ms
step:837/1770 train_time:81780ms step_avg:97.71ms
step:838/1770 train_time:81880ms step_avg:97.71ms
step:839/1770 train_time:81979ms step_avg:97.71ms
step:840/1770 train_time:82079ms step_avg:97.71ms
step:841/1770 train_time:82179ms step_avg:97.72ms
step:842/1770 train_time:82279ms step_avg:97.72ms
step:843/1770 train_time:82379ms step_avg:97.72ms
step:844/1770 train_time:82479ms step_avg:97.72ms
step:845/1770 train_time:82579ms step_avg:97.73ms
step:846/1770 train_time:82679ms step_avg:97.73ms
step:847/1770 train_time:82780ms step_avg:97.73ms
step:848/1770 train_time:82880ms step_avg:97.74ms
step:849/1770 train_time:82980ms step_avg:97.74ms
step:850/1770 train_time:83080ms step_avg:97.74ms
step:851/1770 train_time:83180ms step_avg:97.74ms
step:852/1770 train_time:83280ms step_avg:97.75ms
step:853/1770 train_time:83380ms step_avg:97.75ms
step:854/1770 train_time:83480ms step_avg:97.75ms
step:855/1770 train_time:83580ms step_avg:97.75ms
step:856/1770 train_time:83680ms step_avg:97.76ms
step:857/1770 train_time:83780ms step_avg:97.76ms
step:858/1770 train_time:83880ms step_avg:97.76ms
step:859/1770 train_time:83980ms step_avg:97.76ms
step:860/1770 train_time:84080ms step_avg:97.77ms
step:861/1770 train_time:84180ms step_avg:97.77ms
step:862/1770 train_time:84280ms step_avg:97.77ms
step:863/1770 train_time:84380ms step_avg:97.78ms
step:864/1770 train_time:84480ms step_avg:97.78ms
step:865/1770 train_time:84580ms step_avg:97.78ms
step:866/1770 train_time:84680ms step_avg:97.78ms
step:867/1770 train_time:84779ms step_avg:97.78ms
step:868/1770 train_time:84879ms step_avg:97.79ms
step:869/1770 train_time:84979ms step_avg:97.79ms
step:870/1770 train_time:85080ms step_avg:97.79ms
step:871/1770 train_time:85180ms step_avg:97.80ms
step:872/1770 train_time:85280ms step_avg:97.80ms
step:873/1770 train_time:85379ms step_avg:97.80ms
step:874/1770 train_time:85479ms step_avg:97.80ms
step:875/1770 train_time:85579ms step_avg:97.81ms
step:875/1770 val_loss:3.5491 train_time:85676ms step_avg:97.92ms
step:876/1770 train_time:85703ms step_avg:97.83ms
step:877/1770 train_time:85786ms step_avg:97.82ms
step:878/1770 train_time:85889ms step_avg:97.82ms
step:879/1770 train_time:85988ms step_avg:97.83ms
step:880/1770 train_time:86088ms step_avg:97.83ms
step:881/1770 train_time:86189ms step_avg:97.83ms
step:882/1770 train_time:86288ms step_avg:97.83ms
step:883/1770 train_time:86388ms step_avg:97.83ms
step:884/1770 train_time:86487ms step_avg:97.84ms
step:885/1770 train_time:86588ms step_avg:97.84ms
step:886/1770 train_time:86689ms step_avg:97.84ms
step:887/1770 train_time:86791ms step_avg:97.85ms
step:888/1770 train_time:86892ms step_avg:97.85ms
step:889/1770 train_time:86992ms step_avg:97.85ms
step:890/1770 train_time:87091ms step_avg:97.86ms
step:891/1770 train_time:87192ms step_avg:97.86ms
step:892/1770 train_time:87292ms step_avg:97.86ms
step:893/1770 train_time:87391ms step_avg:97.86ms
step:894/1770 train_time:87491ms step_avg:97.86ms
step:895/1770 train_time:87591ms step_avg:97.87ms
step:896/1770 train_time:87692ms step_avg:97.87ms
step:897/1770 train_time:87793ms step_avg:97.87ms
step:898/1770 train_time:87894ms step_avg:97.88ms
step:899/1770 train_time:87994ms step_avg:97.88ms
step:900/1770 train_time:88095ms step_avg:97.88ms
step:901/1770 train_time:88197ms step_avg:97.89ms
step:902/1770 train_time:88297ms step_avg:97.89ms
step:903/1770 train_time:88398ms step_avg:97.89ms
step:904/1770 train_time:88499ms step_avg:97.90ms
step:905/1770 train_time:88601ms step_avg:97.90ms
step:906/1770 train_time:88702ms step_avg:97.91ms
step:907/1770 train_time:88804ms step_avg:97.91ms
step:908/1770 train_time:88906ms step_avg:97.91ms
step:909/1770 train_time:89007ms step_avg:97.92ms
step:910/1770 train_time:89107ms step_avg:97.92ms
step:911/1770 train_time:89208ms step_avg:97.92ms
step:912/1770 train_time:89308ms step_avg:97.93ms
step:913/1770 train_time:89408ms step_avg:97.93ms
step:914/1770 train_time:89508ms step_avg:97.93ms
step:915/1770 train_time:89607ms step_avg:97.93ms
step:916/1770 train_time:89707ms step_avg:97.93ms
step:917/1770 train_time:89808ms step_avg:97.94ms
step:918/1770 train_time:89908ms step_avg:97.94ms
step:919/1770 train_time:90008ms step_avg:97.94ms
step:920/1770 train_time:90110ms step_avg:97.95ms
step:921/1770 train_time:90212ms step_avg:97.95ms
step:922/1770 train_time:90314ms step_avg:97.95ms
step:923/1770 train_time:90415ms step_avg:97.96ms
step:924/1770 train_time:90517ms step_avg:97.96ms
step:925/1770 train_time:90619ms step_avg:97.97ms
step:926/1770 train_time:90722ms step_avg:97.97ms
step:927/1770 train_time:90825ms step_avg:97.98ms
step:928/1770 train_time:90927ms step_avg:97.98ms
step:929/1770 train_time:91029ms step_avg:97.99ms
step:930/1770 train_time:91130ms step_avg:97.99ms
step:931/1770 train_time:91231ms step_avg:97.99ms
step:932/1770 train_time:91333ms step_avg:98.00ms
step:933/1770 train_time:91434ms step_avg:98.00ms
step:934/1770 train_time:91535ms step_avg:98.00ms
step:935/1770 train_time:91638ms step_avg:98.01ms
step:936/1770 train_time:91740ms step_avg:98.01ms
step:937/1770 train_time:91843ms step_avg:98.02ms
step:938/1770 train_time:91947ms step_avg:98.02ms
step:939/1770 train_time:92049ms step_avg:98.03ms
step:940/1770 train_time:92150ms step_avg:98.03ms
step:941/1770 train_time:92252ms step_avg:98.04ms
step:942/1770 train_time:92353ms step_avg:98.04ms
step:943/1770 train_time:92455ms step_avg:98.04ms
step:944/1770 train_time:92557ms step_avg:98.05ms
step:945/1770 train_time:92658ms step_avg:98.05ms
step:946/1770 train_time:92760ms step_avg:98.05ms
step:947/1770 train_time:92863ms step_avg:98.06ms
step:948/1770 train_time:92967ms step_avg:98.07ms
step:949/1770 train_time:93070ms step_avg:98.07ms
step:950/1770 train_time:93172ms step_avg:98.08ms
step:951/1770 train_time:93274ms step_avg:98.08ms
step:952/1770 train_time:93376ms step_avg:98.08ms
step:953/1770 train_time:93476ms step_avg:98.09ms
step:954/1770 train_time:93578ms step_avg:98.09ms
step:955/1770 train_time:93680ms step_avg:98.09ms
step:956/1770 train_time:93782ms step_avg:98.10ms
step:957/1770 train_time:93886ms step_avg:98.10ms
step:958/1770 train_time:93988ms step_avg:98.11ms
step:959/1770 train_time:94091ms step_avg:98.11ms
step:960/1770 train_time:94194ms step_avg:98.12ms
step:961/1770 train_time:94296ms step_avg:98.12ms
step:962/1770 train_time:94398ms step_avg:98.13ms
step:963/1770 train_time:94500ms step_avg:98.13ms
step:964/1770 train_time:94603ms step_avg:98.14ms
step:965/1770 train_time:94704ms step_avg:98.14ms
step:966/1770 train_time:94806ms step_avg:98.14ms
step:967/1770 train_time:94909ms step_avg:98.15ms
step:968/1770 train_time:95011ms step_avg:98.15ms
step:969/1770 train_time:95113ms step_avg:98.16ms
step:970/1770 train_time:95214ms step_avg:98.16ms
step:971/1770 train_time:95316ms step_avg:98.16ms
step:972/1770 train_time:95418ms step_avg:98.17ms
step:973/1770 train_time:95519ms step_avg:98.17ms
step:974/1770 train_time:95622ms step_avg:98.17ms
step:975/1770 train_time:95725ms step_avg:98.18ms
step:976/1770 train_time:95828ms step_avg:98.18ms
step:977/1770 train_time:95930ms step_avg:98.19ms
step:978/1770 train_time:96032ms step_avg:98.19ms
step:979/1770 train_time:96134ms step_avg:98.20ms
step:980/1770 train_time:96236ms step_avg:98.20ms
step:981/1770 train_time:96337ms step_avg:98.20ms
step:982/1770 train_time:96439ms step_avg:98.21ms
step:983/1770 train_time:96542ms step_avg:98.21ms
step:984/1770 train_time:96645ms step_avg:98.22ms
step:985/1770 train_time:96748ms step_avg:98.22ms
step:986/1770 train_time:96849ms step_avg:98.22ms
step:987/1770 train_time:96951ms step_avg:98.23ms
step:988/1770 train_time:97052ms step_avg:98.23ms
step:989/1770 train_time:97154ms step_avg:98.23ms
step:990/1770 train_time:97255ms step_avg:98.24ms
step:991/1770 train_time:97357ms step_avg:98.24ms
step:992/1770 train_time:97458ms step_avg:98.24ms
step:993/1770 train_time:97560ms step_avg:98.25ms
step:994/1770 train_time:97663ms step_avg:98.25ms
step:995/1770 train_time:97767ms step_avg:98.26ms
step:996/1770 train_time:97870ms step_avg:98.26ms
step:997/1770 train_time:97973ms step_avg:98.27ms
step:998/1770 train_time:98074ms step_avg:98.27ms
step:999/1770 train_time:98176ms step_avg:98.27ms
step:1000/1770 train_time:98278ms step_avg:98.28ms
step:1000/1770 val_loss:3.5108 train_time:98377ms step_avg:98.38ms
step:1001/1770 train_time:98403ms step_avg:98.31ms
step:1002/1770 train_time:98490ms step_avg:98.29ms
step:1003/1770 train_time:98594ms step_avg:98.30ms
step:1004/1770 train_time:98696ms step_avg:98.30ms
step:1005/1770 train_time:98797ms step_avg:98.31ms
step:1006/1770 train_time:98898ms step_avg:98.31ms
step:1007/1770 train_time:98999ms step_avg:98.31ms
step:1008/1770 train_time:99100ms step_avg:98.31ms
step:1009/1770 train_time:99202ms step_avg:98.32ms
step:1010/1770 train_time:99303ms step_avg:98.32ms
step:1011/1770 train_time:99407ms step_avg:98.33ms
step:1012/1770 train_time:99511ms step_avg:98.33ms
step:1013/1770 train_time:99614ms step_avg:98.34ms
step:1014/1770 train_time:99715ms step_avg:98.34ms
step:1015/1770 train_time:99817ms step_avg:98.34ms
step:1016/1770 train_time:99919ms step_avg:98.35ms
step:1017/1770 train_time:100019ms step_avg:98.35ms
step:1018/1770 train_time:100120ms step_avg:98.35ms
step:1019/1770 train_time:100221ms step_avg:98.35ms
step:1020/1770 train_time:100323ms step_avg:98.36ms
step:1021/1770 train_time:100427ms step_avg:98.36ms
step:1022/1770 train_time:100531ms step_avg:98.37ms
step:1023/1770 train_time:100633ms step_avg:98.37ms
step:1024/1770 train_time:100735ms step_avg:98.37ms
step:1025/1770 train_time:100837ms step_avg:98.38ms
step:1026/1770 train_time:100939ms step_avg:98.38ms
step:1027/1770 train_time:101041ms step_avg:98.38ms
step:1028/1770 train_time:101142ms step_avg:98.39ms
step:1029/1770 train_time:101243ms step_avg:98.39ms
step:1030/1770 train_time:101345ms step_avg:98.39ms
step:1031/1770 train_time:101449ms step_avg:98.40ms
step:1032/1770 train_time:101553ms step_avg:98.40ms
step:1033/1770 train_time:101656ms step_avg:98.41ms
step:1034/1770 train_time:101759ms step_avg:98.41ms
step:1035/1770 train_time:101861ms step_avg:98.42ms
step:1036/1770 train_time:101962ms step_avg:98.42ms
step:1037/1770 train_time:102064ms step_avg:98.42ms
step:1038/1770 train_time:102166ms step_avg:98.43ms
step:1039/1770 train_time:102268ms step_avg:98.43ms
step:1040/1770 train_time:102370ms step_avg:98.43ms
step:1041/1770 train_time:102473ms step_avg:98.44ms
step:1042/1770 train_time:102575ms step_avg:98.44ms
step:1043/1770 train_time:102677ms step_avg:98.44ms
step:1044/1770 train_time:102779ms step_avg:98.45ms
step:1045/1770 train_time:102881ms step_avg:98.45ms
step:1046/1770 train_time:102982ms step_avg:98.45ms
step:1047/1770 train_time:103083ms step_avg:98.46ms
step:1048/1770 train_time:103186ms step_avg:98.46ms
step:1049/1770 train_time:103288ms step_avg:98.46ms
step:1050/1770 train_time:103391ms step_avg:98.47ms
step:1051/1770 train_time:103495ms step_avg:98.47ms
step:1052/1770 train_time:103597ms step_avg:98.48ms
step:1053/1770 train_time:103699ms step_avg:98.48ms
step:1054/1770 train_time:103802ms step_avg:98.48ms
step:1055/1770 train_time:103905ms step_avg:98.49ms
step:1056/1770 train_time:104007ms step_avg:98.49ms
step:1057/1770 train_time:104109ms step_avg:98.49ms
step:1058/1770 train_time:104211ms step_avg:98.50ms
step:1059/1770 train_time:104314ms step_avg:98.50ms
step:1060/1770 train_time:104416ms step_avg:98.51ms
step:1061/1770 train_time:104518ms step_avg:98.51ms
step:1062/1770 train_time:104620ms step_avg:98.51ms
step:1063/1770 train_time:104724ms step_avg:98.52ms
step:1064/1770 train_time:104827ms step_avg:98.52ms
step:1065/1770 train_time:104931ms step_avg:98.53ms
step:1066/1770 train_time:105033ms step_avg:98.53ms
step:1067/1770 train_time:105135ms step_avg:98.53ms
step:1068/1770 train_time:105237ms step_avg:98.54ms
step:1069/1770 train_time:105340ms step_avg:98.54ms
step:1070/1770 train_time:105442ms step_avg:98.54ms
step:1071/1770 train_time:105545ms step_avg:98.55ms
step:1072/1770 train_time:105648ms step_avg:98.55ms
step:1073/1770 train_time:105751ms step_avg:98.56ms
step:1074/1770 train_time:105853ms step_avg:98.56ms
step:1075/1770 train_time:105955ms step_avg:98.56ms
step:1076/1770 train_time:106057ms step_avg:98.57ms
step:1077/1770 train_time:106159ms step_avg:98.57ms
step:1078/1770 train_time:106261ms step_avg:98.57ms
step:1079/1770 train_time:106363ms step_avg:98.58ms
step:1080/1770 train_time:106466ms step_avg:98.58ms
step:1081/1770 train_time:106570ms step_avg:98.58ms
step:1082/1770 train_time:106672ms step_avg:98.59ms
step:1083/1770 train_time:106774ms step_avg:98.59ms
step:1084/1770 train_time:106876ms step_avg:98.59ms
step:1085/1770 train_time:106978ms step_avg:98.60ms
step:1086/1770 train_time:107079ms step_avg:98.60ms
step:1087/1770 train_time:107181ms step_avg:98.60ms
step:1088/1770 train_time:107282ms step_avg:98.61ms
step:1089/1770 train_time:107385ms step_avg:98.61ms
step:1090/1770 train_time:107487ms step_avg:98.61ms
step:1091/1770 train_time:107590ms step_avg:98.62ms
step:1092/1770 train_time:107694ms step_avg:98.62ms
step:1093/1770 train_time:107796ms step_avg:98.62ms
step:1094/1770 train_time:107898ms step_avg:98.63ms
step:1095/1770 train_time:108000ms step_avg:98.63ms
step:1096/1770 train_time:108102ms step_avg:98.63ms
step:1097/1770 train_time:108204ms step_avg:98.64ms
step:1098/1770 train_time:108307ms step_avg:98.64ms
step:1099/1770 train_time:108411ms step_avg:98.65ms
step:1100/1770 train_time:108513ms step_avg:98.65ms
step:1101/1770 train_time:108616ms step_avg:98.65ms
step:1102/1770 train_time:108718ms step_avg:98.66ms
step:1103/1770 train_time:108820ms step_avg:98.66ms
step:1104/1770 train_time:108923ms step_avg:98.66ms
step:1105/1770 train_time:109026ms step_avg:98.67ms
step:1106/1770 train_time:109130ms step_avg:98.67ms
step:1107/1770 train_time:109232ms step_avg:98.67ms
step:1108/1770 train_time:109334ms step_avg:98.68ms
step:1109/1770 train_time:109436ms step_avg:98.68ms
step:1110/1770 train_time:109538ms step_avg:98.68ms
step:1111/1770 train_time:109641ms step_avg:98.69ms
step:1112/1770 train_time:109743ms step_avg:98.69ms
step:1113/1770 train_time:109845ms step_avg:98.69ms
step:1114/1770 train_time:109948ms step_avg:98.70ms
step:1115/1770 train_time:110052ms step_avg:98.70ms
step:1116/1770 train_time:110154ms step_avg:98.70ms
step:1117/1770 train_time:110257ms step_avg:98.71ms
step:1118/1770 train_time:110358ms step_avg:98.71ms
step:1119/1770 train_time:110460ms step_avg:98.71ms
step:1120/1770 train_time:110562ms step_avg:98.72ms
step:1121/1770 train_time:110664ms step_avg:98.72ms
step:1122/1770 train_time:110766ms step_avg:98.72ms
step:1123/1770 train_time:110869ms step_avg:98.73ms
step:1124/1770 train_time:110972ms step_avg:98.73ms
step:1125/1770 train_time:111075ms step_avg:98.73ms
step:1125/1770 val_loss:3.4727 train_time:111174ms step_avg:98.82ms
step:1126/1770 train_time:111201ms step_avg:98.76ms
step:1127/1770 train_time:111291ms step_avg:98.75ms
step:1128/1770 train_time:111394ms step_avg:98.75ms
step:1129/1770 train_time:111495ms step_avg:98.76ms
step:1130/1770 train_time:111597ms step_avg:98.76ms
step:1131/1770 train_time:111698ms step_avg:98.76ms
step:1132/1770 train_time:111799ms step_avg:98.76ms
step:1133/1770 train_time:111900ms step_avg:98.76ms
step:1134/1770 train_time:112001ms step_avg:98.77ms
step:1135/1770 train_time:112103ms step_avg:98.77ms
step:1136/1770 train_time:112208ms step_avg:98.77ms
step:1137/1770 train_time:112313ms step_avg:98.78ms
step:1138/1770 train_time:112417ms step_avg:98.78ms
step:1139/1770 train_time:112521ms step_avg:98.79ms
step:1140/1770 train_time:112624ms step_avg:98.79ms
step:1141/1770 train_time:112727ms step_avg:98.80ms
step:1142/1770 train_time:112829ms step_avg:98.80ms
step:1143/1770 train_time:112931ms step_avg:98.80ms
step:1144/1770 train_time:113032ms step_avg:98.80ms
step:1145/1770 train_time:113134ms step_avg:98.81ms
step:1146/1770 train_time:113237ms step_avg:98.81ms
step:1147/1770 train_time:113340ms step_avg:98.81ms
step:1148/1770 train_time:113444ms step_avg:98.82ms
step:1149/1770 train_time:113549ms step_avg:98.82ms
step:1150/1770 train_time:113651ms step_avg:98.83ms
step:1151/1770 train_time:113754ms step_avg:98.83ms
step:1152/1770 train_time:113855ms step_avg:98.83ms
step:1153/1770 train_time:113957ms step_avg:98.83ms
step:1154/1770 train_time:114058ms step_avg:98.84ms
step:1155/1770 train_time:114160ms step_avg:98.84ms
step:1156/1770 train_time:114262ms step_avg:98.84ms
step:1157/1770 train_time:114366ms step_avg:98.85ms
step:1158/1770 train_time:114470ms step_avg:98.85ms
step:1159/1770 train_time:114573ms step_avg:98.85ms
step:1160/1770 train_time:114675ms step_avg:98.86ms
step:1161/1770 train_time:114776ms step_avg:98.86ms
step:1162/1770 train_time:114878ms step_avg:98.86ms
step:1163/1770 train_time:114980ms step_avg:98.86ms
step:1164/1770 train_time:115081ms step_avg:98.87ms
step:1165/1770 train_time:115185ms step_avg:98.87ms
step:1166/1770 train_time:115289ms step_avg:98.88ms
step:1167/1770 train_time:115392ms step_avg:98.88ms
step:1168/1770 train_time:115495ms step_avg:98.88ms
step:1169/1770 train_time:115597ms step_avg:98.89ms
step:1170/1770 train_time:115699ms step_avg:98.89ms
step:1171/1770 train_time:115801ms step_avg:98.89ms
step:1172/1770 train_time:115904ms step_avg:98.89ms
step:1173/1770 train_time:116006ms step_avg:98.90ms
step:1174/1770 train_time:116109ms step_avg:98.90ms
step:1175/1770 train_time:116211ms step_avg:98.90ms
step:1176/1770 train_time:116314ms step_avg:98.91ms
step:1177/1770 train_time:116415ms step_avg:98.91ms
step:1178/1770 train_time:116518ms step_avg:98.91ms
step:1179/1770 train_time:116621ms step_avg:98.92ms
step:1180/1770 train_time:116724ms step_avg:98.92ms
step:1181/1770 train_time:116827ms step_avg:98.92ms
step:1182/1770 train_time:116930ms step_avg:98.93ms
step:1183/1770 train_time:117034ms step_avg:98.93ms
step:1184/1770 train_time:117138ms step_avg:98.93ms
step:1185/1770 train_time:117241ms step_avg:98.94ms
step:1186/1770 train_time:117345ms step_avg:98.94ms
step:1187/1770 train_time:117449ms step_avg:98.95ms
step:1188/1770 train_time:117554ms step_avg:98.95ms
step:1189/1770 train_time:117657ms step_avg:98.95ms
step:1190/1770 train_time:117761ms step_avg:98.96ms
step:1191/1770 train_time:117864ms step_avg:98.96ms
step:1192/1770 train_time:117969ms step_avg:98.97ms
step:1193/1770 train_time:118072ms step_avg:98.97ms
step:1194/1770 train_time:118175ms step_avg:98.97ms
step:1195/1770 train_time:118277ms step_avg:98.98ms
step:1196/1770 train_time:118382ms step_avg:98.98ms
step:1197/1770 train_time:118486ms step_avg:98.99ms
step:1198/1770 train_time:118591ms step_avg:98.99ms
step:1199/1770 train_time:118694ms step_avg:98.99ms
step:1200/1770 train_time:118797ms step_avg:99.00ms
step:1201/1770 train_time:118900ms step_avg:99.00ms
step:1202/1770 train_time:119005ms step_avg:99.01ms
step:1203/1770 train_time:119109ms step_avg:99.01ms
step:1204/1770 train_time:119213ms step_avg:99.01ms
step:1205/1770 train_time:119316ms step_avg:99.02ms
step:1206/1770 train_time:119420ms step_avg:99.02ms
step:1207/1770 train_time:119525ms step_avg:99.03ms
step:1208/1770 train_time:119631ms step_avg:99.03ms
step:1209/1770 train_time:119735ms step_avg:99.04ms
step:1210/1770 train_time:119837ms step_avg:99.04ms
step:1211/1770 train_time:119941ms step_avg:99.04ms
step:1212/1770 train_time:120046ms step_avg:99.05ms
step:1213/1770 train_time:120150ms step_avg:99.05ms
step:1214/1770 train_time:120254ms step_avg:99.06ms
step:1215/1770 train_time:120356ms step_avg:99.06ms
step:1216/1770 train_time:120461ms step_avg:99.06ms
step:1217/1770 train_time:120565ms step_avg:99.07ms
step:1218/1770 train_time:120669ms step_avg:99.07ms
step:1219/1770 train_time:120773ms step_avg:99.08ms
step:1220/1770 train_time:120877ms step_avg:99.08ms
step:1221/1770 train_time:120980ms step_avg:99.08ms
step:1222/1770 train_time:121085ms step_avg:99.09ms
step:1223/1770 train_time:121189ms step_avg:99.09ms
step:1224/1770 train_time:121294ms step_avg:99.10ms
step:1225/1770 train_time:121398ms step_avg:99.10ms
step:1226/1770 train_time:121501ms step_avg:99.10ms
step:1227/1770 train_time:121605ms step_avg:99.11ms
step:1228/1770 train_time:121710ms step_avg:99.11ms
step:1229/1770 train_time:121813ms step_avg:99.12ms
step:1230/1770 train_time:121916ms step_avg:99.12ms
step:1231/1770 train_time:122019ms step_avg:99.12ms
step:1232/1770 train_time:122123ms step_avg:99.13ms
step:1233/1770 train_time:122227ms step_avg:99.13ms
step:1234/1770 train_time:122331ms step_avg:99.13ms
step:1235/1770 train_time:122435ms step_avg:99.14ms
step:1236/1770 train_time:122539ms step_avg:99.14ms
step:1237/1770 train_time:122643ms step_avg:99.15ms
step:1238/1770 train_time:122747ms step_avg:99.15ms
step:1239/1770 train_time:122851ms step_avg:99.15ms
step:1240/1770 train_time:122955ms step_avg:99.16ms
step:1241/1770 train_time:123059ms step_avg:99.16ms
step:1242/1770 train_time:123162ms step_avg:99.16ms
step:1243/1770 train_time:123266ms step_avg:99.17ms
step:1244/1770 train_time:123371ms step_avg:99.17ms
step:1245/1770 train_time:123474ms step_avg:99.18ms
step:1246/1770 train_time:123578ms step_avg:99.18ms
step:1247/1770 train_time:123681ms step_avg:99.18ms
step:1248/1770 train_time:123785ms step_avg:99.19ms
step:1249/1770 train_time:123890ms step_avg:99.19ms
step:1250/1770 train_time:123993ms step_avg:99.19ms
step:1250/1770 val_loss:3.4234 train_time:124094ms step_avg:99.28ms
step:1251/1770 train_time:124121ms step_avg:99.22ms
step:1252/1770 train_time:124209ms step_avg:99.21ms
step:1253/1770 train_time:124312ms step_avg:99.21ms
step:1254/1770 train_time:124415ms step_avg:99.21ms
step:1255/1770 train_time:124519ms step_avg:99.22ms
step:1256/1770 train_time:124622ms step_avg:99.22ms
step:1257/1770 train_time:124724ms step_avg:99.22ms
step:1258/1770 train_time:124827ms step_avg:99.23ms
step:1259/1770 train_time:124930ms step_avg:99.23ms
step:1260/1770 train_time:125033ms step_avg:99.23ms
step:1261/1770 train_time:125139ms step_avg:99.24ms
step:1262/1770 train_time:125244ms step_avg:99.24ms
step:1263/1770 train_time:125349ms step_avg:99.25ms
step:1264/1770 train_time:125453ms step_avg:99.25ms
step:1265/1770 train_time:125556ms step_avg:99.25ms
step:1266/1770 train_time:125661ms step_avg:99.26ms
step:1267/1770 train_time:125764ms step_avg:99.26ms
step:1268/1770 train_time:125867ms step_avg:99.26ms
step:1269/1770 train_time:125970ms step_avg:99.27ms
step:1270/1770 train_time:126074ms step_avg:99.27ms
step:1271/1770 train_time:126180ms step_avg:99.28ms
step:1272/1770 train_time:126284ms step_avg:99.28ms
step:1273/1770 train_time:126387ms step_avg:99.28ms
step:1274/1770 train_time:126490ms step_avg:99.29ms
step:1275/1770 train_time:126594ms step_avg:99.29ms
step:1276/1770 train_time:126698ms step_avg:99.29ms
step:1277/1770 train_time:126801ms step_avg:99.30ms
step:1278/1770 train_time:126905ms step_avg:99.30ms
step:1279/1770 train_time:127008ms step_avg:99.30ms
step:1280/1770 train_time:127112ms step_avg:99.31ms
step:1281/1770 train_time:127216ms step_avg:99.31ms
step:1282/1770 train_time:127320ms step_avg:99.31ms
step:1283/1770 train_time:127424ms step_avg:99.32ms
step:1284/1770 train_time:127527ms step_avg:99.32ms
step:1285/1770 train_time:127629ms step_avg:99.32ms
step:1286/1770 train_time:127732ms step_avg:99.33ms
step:1287/1770 train_time:127836ms step_avg:99.33ms
step:1288/1770 train_time:127940ms step_avg:99.33ms
step:1289/1770 train_time:128044ms step_avg:99.34ms
step:1290/1770 train_time:128147ms step_avg:99.34ms
step:1291/1770 train_time:128249ms step_avg:99.34ms
step:1292/1770 train_time:128354ms step_avg:99.35ms
step:1293/1770 train_time:128461ms step_avg:99.35ms
step:1294/1770 train_time:128564ms step_avg:99.35ms
step:1295/1770 train_time:128667ms step_avg:99.36ms
step:1296/1770 train_time:128770ms step_avg:99.36ms
step:1297/1770 train_time:128873ms step_avg:99.36ms
step:1298/1770 train_time:128978ms step_avg:99.37ms
step:1299/1770 train_time:129082ms step_avg:99.37ms
step:1300/1770 train_time:129185ms step_avg:99.37ms
step:1301/1770 train_time:129289ms step_avg:99.38ms
step:1302/1770 train_time:129391ms step_avg:99.38ms
step:1303/1770 train_time:129496ms step_avg:99.38ms
step:1304/1770 train_time:129601ms step_avg:99.39ms
step:1305/1770 train_time:129704ms step_avg:99.39ms
step:1306/1770 train_time:129807ms step_avg:99.39ms
step:1307/1770 train_time:129911ms step_avg:99.40ms
step:1308/1770 train_time:130014ms step_avg:99.40ms
step:1309/1770 train_time:130120ms step_avg:99.40ms
step:1310/1770 train_time:130223ms step_avg:99.41ms
step:1311/1770 train_time:130327ms step_avg:99.41ms
step:1312/1770 train_time:130430ms step_avg:99.41ms
step:1313/1770 train_time:130532ms step_avg:99.42ms
step:1314/1770 train_time:130638ms step_avg:99.42ms
step:1315/1770 train_time:130741ms step_avg:99.42ms
step:1316/1770 train_time:130845ms step_avg:99.43ms
step:1317/1770 train_time:130948ms step_avg:99.43ms
step:1318/1770 train_time:131053ms step_avg:99.43ms
step:1319/1770 train_time:131158ms step_avg:99.44ms
step:1320/1770 train_time:131262ms step_avg:99.44ms
step:1321/1770 train_time:131366ms step_avg:99.44ms
step:1322/1770 train_time:131470ms step_avg:99.45ms
step:1323/1770 train_time:131573ms step_avg:99.45ms
step:1324/1770 train_time:131677ms step_avg:99.45ms
step:1325/1770 train_time:131783ms step_avg:99.46ms
step:1326/1770 train_time:131886ms step_avg:99.46ms
step:1327/1770 train_time:131992ms step_avg:99.47ms
step:1328/1770 train_time:132096ms step_avg:99.47ms
step:1329/1770 train_time:132200ms step_avg:99.47ms
step:1330/1770 train_time:132305ms step_avg:99.48ms
step:1331/1770 train_time:132407ms step_avg:99.48ms
step:1332/1770 train_time:132510ms step_avg:99.48ms
step:1333/1770 train_time:132613ms step_avg:99.48ms
step:1334/1770 train_time:132718ms step_avg:99.49ms
step:1335/1770 train_time:132823ms step_avg:99.49ms
step:1336/1770 train_time:132926ms step_avg:99.50ms
step:1337/1770 train_time:133029ms step_avg:99.50ms
step:1338/1770 train_time:133133ms step_avg:99.50ms
step:1339/1770 train_time:133238ms step_avg:99.51ms
step:1340/1770 train_time:133343ms step_avg:99.51ms
step:1341/1770 train_time:133446ms step_avg:99.51ms
step:1342/1770 train_time:133549ms step_avg:99.51ms
step:1343/1770 train_time:133653ms step_avg:99.52ms
step:1344/1770 train_time:133758ms step_avg:99.52ms
step:1345/1770 train_time:133861ms step_avg:99.53ms
step:1346/1770 train_time:133965ms step_avg:99.53ms
step:1347/1770 train_time:134069ms step_avg:99.53ms
step:1348/1770 train_time:134176ms step_avg:99.54ms
step:1349/1770 train_time:134280ms step_avg:99.54ms
step:1350/1770 train_time:134384ms step_avg:99.54ms
step:1351/1770 train_time:134486ms step_avg:99.55ms
step:1352/1770 train_time:134589ms step_avg:99.55ms
step:1353/1770 train_time:134693ms step_avg:99.55ms
step:1354/1770 train_time:134797ms step_avg:99.55ms
step:1355/1770 train_time:134902ms step_avg:99.56ms
step:1356/1770 train_time:135005ms step_avg:99.56ms
step:1357/1770 train_time:135109ms step_avg:99.56ms
step:1358/1770 train_time:135213ms step_avg:99.57ms
step:1359/1770 train_time:135318ms step_avg:99.57ms
step:1360/1770 train_time:135424ms step_avg:99.58ms
step:1361/1770 train_time:135527ms step_avg:99.58ms
step:1362/1770 train_time:135630ms step_avg:99.58ms
step:1363/1770 train_time:135733ms step_avg:99.58ms
step:1364/1770 train_time:135838ms step_avg:99.59ms
step:1365/1770 train_time:135942ms step_avg:99.59ms
step:1366/1770 train_time:136046ms step_avg:99.59ms
step:1367/1770 train_time:136149ms step_avg:99.60ms
step:1368/1770 train_time:136252ms step_avg:99.60ms
step:1369/1770 train_time:136357ms step_avg:99.60ms
step:1370/1770 train_time:136464ms step_avg:99.61ms
step:1371/1770 train_time:136567ms step_avg:99.61ms
step:1372/1770 train_time:136671ms step_avg:99.61ms
step:1373/1770 train_time:136775ms step_avg:99.62ms
step:1374/1770 train_time:136881ms step_avg:99.62ms
step:1375/1770 train_time:136985ms step_avg:99.63ms
step:1375/1770 val_loss:3.3798 train_time:137086ms step_avg:99.70ms
step:1376/1770 train_time:137113ms step_avg:99.65ms
step:1377/1770 train_time:137201ms step_avg:99.64ms
step:1378/1770 train_time:137307ms step_avg:99.64ms
step:1379/1770 train_time:137411ms step_avg:99.65ms
step:1380/1770 train_time:137515ms step_avg:99.65ms
step:1381/1770 train_time:137618ms step_avg:99.65ms
step:1382/1770 train_time:137721ms step_avg:99.65ms
step:1383/1770 train_time:137824ms step_avg:99.66ms
step:1384/1770 train_time:137926ms step_avg:99.66ms
step:1385/1770 train_time:138031ms step_avg:99.66ms
step:1386/1770 train_time:138137ms step_avg:99.67ms
step:1387/1770 train_time:138243ms step_avg:99.67ms
step:1388/1770 train_time:138347ms step_avg:99.67ms
step:1389/1770 train_time:138452ms step_avg:99.68ms
step:1390/1770 train_time:138556ms step_avg:99.68ms
step:1391/1770 train_time:138659ms step_avg:99.68ms
step:1392/1770 train_time:138761ms step_avg:99.68ms
step:1393/1770 train_time:138864ms step_avg:99.69ms
step:1394/1770 train_time:138967ms step_avg:99.69ms
step:1395/1770 train_time:139072ms step_avg:99.69ms
step:1396/1770 train_time:139177ms step_avg:99.70ms
step:1397/1770 train_time:139282ms step_avg:99.70ms
step:1398/1770 train_time:139387ms step_avg:99.70ms
step:1399/1770 train_time:139492ms step_avg:99.71ms
step:1400/1770 train_time:139597ms step_avg:99.71ms
step:1401/1770 train_time:139700ms step_avg:99.71ms
step:1402/1770 train_time:139803ms step_avg:99.72ms
step:1403/1770 train_time:139906ms step_avg:99.72ms
step:1404/1770 train_time:140013ms step_avg:99.72ms
step:1405/1770 train_time:140116ms step_avg:99.73ms
step:1406/1770 train_time:140220ms step_avg:99.73ms
step:1407/1770 train_time:140324ms step_avg:99.73ms
step:1408/1770 train_time:140428ms step_avg:99.74ms
step:1409/1770 train_time:140534ms step_avg:99.74ms
step:1410/1770 train_time:140637ms step_avg:99.74ms
step:1411/1770 train_time:140740ms step_avg:99.74ms
step:1412/1770 train_time:140843ms step_avg:99.75ms
step:1413/1770 train_time:140947ms step_avg:99.75ms
step:1414/1770 train_time:141052ms step_avg:99.75ms
step:1415/1770 train_time:141156ms step_avg:99.76ms
step:1416/1770 train_time:141260ms step_avg:99.76ms
step:1417/1770 train_time:141363ms step_avg:99.76ms
step:1418/1770 train_time:141468ms step_avg:99.77ms
step:1419/1770 train_time:141573ms step_avg:99.77ms
step:1420/1770 train_time:141677ms step_avg:99.77ms
step:1421/1770 train_time:141779ms step_avg:99.77ms
step:1422/1770 train_time:141882ms step_avg:99.78ms
step:1423/1770 train_time:141985ms step_avg:99.78ms
step:1424/1770 train_time:142090ms step_avg:99.78ms
step:1425/1770 train_time:142195ms step_avg:99.79ms
step:1426/1770 train_time:142300ms step_avg:99.79ms
step:1427/1770 train_time:142403ms step_avg:99.79ms
step:1428/1770 train_time:142508ms step_avg:99.80ms
step:1429/1770 train_time:142612ms step_avg:99.80ms
step:1430/1770 train_time:142717ms step_avg:99.80ms
step:1431/1770 train_time:142820ms step_avg:99.80ms
step:1432/1770 train_time:142924ms step_avg:99.81ms
step:1433/1770 train_time:143028ms step_avg:99.81ms
step:1434/1770 train_time:143133ms step_avg:99.81ms
step:1435/1770 train_time:143237ms step_avg:99.82ms
step:1436/1770 train_time:143344ms step_avg:99.82ms
step:1437/1770 train_time:143447ms step_avg:99.82ms
step:1438/1770 train_time:143552ms step_avg:99.83ms
step:1439/1770 train_time:143655ms step_avg:99.83ms
step:1440/1770 train_time:143758ms step_avg:99.83ms
step:1441/1770 train_time:143863ms step_avg:99.84ms
step:1442/1770 train_time:143965ms step_avg:99.84ms
step:1443/1770 train_time:144070ms step_avg:99.84ms
step:1444/1770 train_time:144177ms step_avg:99.85ms
step:1445/1770 train_time:144281ms step_avg:99.85ms
step:1446/1770 train_time:144386ms step_avg:99.85ms
step:1447/1770 train_time:144492ms step_avg:99.86ms
step:1448/1770 train_time:144597ms step_avg:99.86ms
step:1449/1770 train_time:144704ms step_avg:99.86ms
step:1450/1770 train_time:144809ms step_avg:99.87ms
step:1451/1770 train_time:144914ms step_avg:99.87ms
step:1452/1770 train_time:145018ms step_avg:99.87ms
step:1453/1770 train_time:145123ms step_avg:99.88ms
step:1454/1770 train_time:145227ms step_avg:99.88ms
step:1455/1770 train_time:145333ms step_avg:99.89ms
step:1456/1770 train_time:145439ms step_avg:99.89ms
step:1457/1770 train_time:145545ms step_avg:99.89ms
step:1458/1770 train_time:145650ms step_avg:99.90ms
step:1459/1770 train_time:145756ms step_avg:99.90ms
step:1460/1770 train_time:145861ms step_avg:99.90ms
step:1461/1770 train_time:145965ms step_avg:99.91ms
step:1462/1770 train_time:146071ms step_avg:99.91ms
step:1463/1770 train_time:146176ms step_avg:99.92ms
step:1464/1770 train_time:146283ms step_avg:99.92ms
step:1465/1770 train_time:146387ms step_avg:99.92ms
step:1466/1770 train_time:146493ms step_avg:99.93ms
step:1467/1770 train_time:146597ms step_avg:99.93ms
step:1468/1770 train_time:146703ms step_avg:99.93ms
step:1469/1770 train_time:146808ms step_avg:99.94ms
step:1470/1770 train_time:146912ms step_avg:99.94ms
step:1471/1770 train_time:147016ms step_avg:99.94ms
step:1472/1770 train_time:147120ms step_avg:99.95ms
step:1473/1770 train_time:147227ms step_avg:99.95ms
step:1474/1770 train_time:147332ms step_avg:99.95ms
step:1475/1770 train_time:147437ms step_avg:99.96ms
step:1476/1770 train_time:147542ms step_avg:99.96ms
step:1477/1770 train_time:147649ms step_avg:99.97ms
step:1478/1770 train_time:147753ms step_avg:99.97ms
step:1479/1770 train_time:147858ms step_avg:99.97ms
step:1480/1770 train_time:147963ms step_avg:99.97ms
step:1481/1770 train_time:148071ms step_avg:99.98ms
step:1482/1770 train_time:148176ms step_avg:99.98ms
step:1483/1770 train_time:148280ms step_avg:99.99ms
step:1484/1770 train_time:148385ms step_avg:99.99ms
step:1485/1770 train_time:148491ms step_avg:99.99ms
step:1486/1770 train_time:148596ms step_avg:100.00ms
step:1487/1770 train_time:148700ms step_avg:100.00ms
step:1488/1770 train_time:148805ms step_avg:100.00ms
step:1489/1770 train_time:148911ms step_avg:100.01ms
step:1490/1770 train_time:149016ms step_avg:100.01ms
step:1491/1770 train_time:149121ms step_avg:100.01ms
step:1492/1770 train_time:149225ms step_avg:100.02ms
step:1493/1770 train_time:149332ms step_avg:100.02ms
step:1494/1770 train_time:149439ms step_avg:100.03ms
step:1495/1770 train_time:149544ms step_avg:100.03ms
step:1496/1770 train_time:149648ms step_avg:100.03ms
step:1497/1770 train_time:149752ms step_avg:100.04ms
step:1498/1770 train_time:149857ms step_avg:100.04ms
step:1499/1770 train_time:149961ms step_avg:100.04ms
step:1500/1770 train_time:150066ms step_avg:100.04ms
step:1500/1770 val_loss:3.3420 train_time:150170ms step_avg:100.11ms
step:1501/1770 train_time:150197ms step_avg:100.06ms
step:1502/1770 train_time:150288ms step_avg:100.06ms
step:1503/1770 train_time:150394ms step_avg:100.06ms
step:1504/1770 train_time:150498ms step_avg:100.07ms
step:1505/1770 train_time:150604ms step_avg:100.07ms
step:1506/1770 train_time:150708ms step_avg:100.07ms
step:1507/1770 train_time:150813ms step_avg:100.07ms
step:1508/1770 train_time:150917ms step_avg:100.08ms
step:1509/1770 train_time:151021ms step_avg:100.08ms
step:1510/1770 train_time:151125ms step_avg:100.08ms
step:1511/1770 train_time:151234ms step_avg:100.09ms
step:1512/1770 train_time:151341ms step_avg:100.09ms
step:1513/1770 train_time:151447ms step_avg:100.10ms
step:1514/1770 train_time:151552ms step_avg:100.10ms
step:1515/1770 train_time:151656ms step_avg:100.10ms
step:1516/1770 train_time:151761ms step_avg:100.11ms
step:1517/1770 train_time:151865ms step_avg:100.11ms
step:1518/1770 train_time:151971ms step_avg:100.11ms
step:1519/1770 train_time:152075ms step_avg:100.11ms
step:1520/1770 train_time:152180ms step_avg:100.12ms
step:1521/1770 train_time:152286ms step_avg:100.12ms
step:1522/1770 train_time:152392ms step_avg:100.13ms
step:1523/1770 train_time:152499ms step_avg:100.13ms
step:1524/1770 train_time:152604ms step_avg:100.13ms
step:1525/1770 train_time:152709ms step_avg:100.14ms
step:1526/1770 train_time:152813ms step_avg:100.14ms
step:1527/1770 train_time:152917ms step_avg:100.14ms
step:1528/1770 train_time:153023ms step_avg:100.15ms
step:1529/1770 train_time:153129ms step_avg:100.15ms
step:1530/1770 train_time:153236ms step_avg:100.15ms
step:1531/1770 train_time:153340ms step_avg:100.16ms
step:1532/1770 train_time:153445ms step_avg:100.16ms
step:1533/1770 train_time:153551ms step_avg:100.16ms
step:1534/1770 train_time:153655ms step_avg:100.17ms
step:1535/1770 train_time:153760ms step_avg:100.17ms
step:1536/1770 train_time:153866ms step_avg:100.17ms
step:1537/1770 train_time:153971ms step_avg:100.18ms
step:1538/1770 train_time:154078ms step_avg:100.18ms
step:1539/1770 train_time:154182ms step_avg:100.18ms
step:1540/1770 train_time:154291ms step_avg:100.19ms
step:1541/1770 train_time:154397ms step_avg:100.19ms
step:1542/1770 train_time:154503ms step_avg:100.20ms
step:1543/1770 train_time:154608ms step_avg:100.20ms
step:1544/1770 train_time:154715ms step_avg:100.20ms
step:1545/1770 train_time:154820ms step_avg:100.21ms
step:1546/1770 train_time:154926ms step_avg:100.21ms
step:1547/1770 train_time:155032ms step_avg:100.21ms
step:1548/1770 train_time:155137ms step_avg:100.22ms
step:1549/1770 train_time:155242ms step_avg:100.22ms
step:1550/1770 train_time:155347ms step_avg:100.22ms
step:1551/1770 train_time:155452ms step_avg:100.23ms
step:1552/1770 train_time:155559ms step_avg:100.23ms
step:1553/1770 train_time:155664ms step_avg:100.23ms
step:1554/1770 train_time:155768ms step_avg:100.24ms
step:1555/1770 train_time:155873ms step_avg:100.24ms
step:1556/1770 train_time:155978ms step_avg:100.24ms
step:1557/1770 train_time:156083ms step_avg:100.25ms
step:1558/1770 train_time:156187ms step_avg:100.25ms
step:1559/1770 train_time:156293ms step_avg:100.25ms
step:1560/1770 train_time:156398ms step_avg:100.26ms
step:1561/1770 train_time:156505ms step_avg:100.26ms
step:1562/1770 train_time:156610ms step_avg:100.26ms
step:1563/1770 train_time:156715ms step_avg:100.27ms
step:1564/1770 train_time:156819ms step_avg:100.27ms
step:1565/1770 train_time:156923ms step_avg:100.27ms
step:1566/1770 train_time:157027ms step_avg:100.27ms
step:1567/1770 train_time:157133ms step_avg:100.28ms
step:1568/1770 train_time:157238ms step_avg:100.28ms
step:1569/1770 train_time:157343ms step_avg:100.28ms
step:1570/1770 train_time:157449ms step_avg:100.29ms
step:1571/1770 train_time:157555ms step_avg:100.29ms
step:1572/1770 train_time:157661ms step_avg:100.29ms
step:1573/1770 train_time:157768ms step_avg:100.30ms
step:1574/1770 train_time:157873ms step_avg:100.30ms
step:1575/1770 train_time:157978ms step_avg:100.30ms
step:1576/1770 train_time:158082ms step_avg:100.31ms
step:1577/1770 train_time:158187ms step_avg:100.31ms
step:1578/1770 train_time:158294ms step_avg:100.31ms
step:1579/1770 train_time:158398ms step_avg:100.32ms
step:1580/1770 train_time:158504ms step_avg:100.32ms
step:1581/1770 train_time:158613ms step_avg:100.32ms
step:1582/1770 train_time:158719ms step_avg:100.33ms
step:1583/1770 train_time:158824ms step_avg:100.33ms
step:1584/1770 train_time:158931ms step_avg:100.34ms
step:1585/1770 train_time:159037ms step_avg:100.34ms
step:1586/1770 train_time:159144ms step_avg:100.34ms
step:1587/1770 train_time:159249ms step_avg:100.35ms
step:1588/1770 train_time:159355ms step_avg:100.35ms
step:1589/1770 train_time:159461ms step_avg:100.35ms
step:1590/1770 train_time:159566ms step_avg:100.36ms
step:1591/1770 train_time:159671ms step_avg:100.36ms
step:1592/1770 train_time:159777ms step_avg:100.36ms
step:1593/1770 train_time:159881ms step_avg:100.36ms
step:1594/1770 train_time:159985ms step_avg:100.37ms
step:1595/1770 train_time:160090ms step_avg:100.37ms
step:1596/1770 train_time:160196ms step_avg:100.37ms
step:1597/1770 train_time:160299ms step_avg:100.38ms
step:1598/1770 train_time:160404ms step_avg:100.38ms
step:1599/1770 train_time:160511ms step_avg:100.38ms
step:1600/1770 train_time:160617ms step_avg:100.39ms
step:1601/1770 train_time:160722ms step_avg:100.39ms
step:1602/1770 train_time:160829ms step_avg:100.39ms
step:1603/1770 train_time:160933ms step_avg:100.40ms
step:1604/1770 train_time:161037ms step_avg:100.40ms
step:1605/1770 train_time:161142ms step_avg:100.40ms
step:1606/1770 train_time:161247ms step_avg:100.40ms
step:1607/1770 train_time:161357ms step_avg:100.41ms
step:1608/1770 train_time:161462ms step_avg:100.41ms
step:1609/1770 train_time:161568ms step_avg:100.42ms
step:1610/1770 train_time:161674ms step_avg:100.42ms
step:1611/1770 train_time:161781ms step_avg:100.42ms
step:1612/1770 train_time:161886ms step_avg:100.43ms
step:1613/1770 train_time:161992ms step_avg:100.43ms
step:1614/1770 train_time:162097ms step_avg:100.43ms
step:1615/1770 train_time:162203ms step_avg:100.44ms
step:1616/1770 train_time:162308ms step_avg:100.44ms
step:1617/1770 train_time:162413ms step_avg:100.44ms
step:1618/1770 train_time:162519ms step_avg:100.44ms
step:1619/1770 train_time:162624ms step_avg:100.45ms
step:1620/1770 train_time:162733ms step_avg:100.45ms
step:1621/1770 train_time:162837ms step_avg:100.45ms
step:1622/1770 train_time:162942ms step_avg:100.46ms
step:1623/1770 train_time:163047ms step_avg:100.46ms
step:1624/1770 train_time:163152ms step_avg:100.46ms
step:1625/1770 train_time:163256ms step_avg:100.47ms
step:1625/1770 val_loss:3.3079 train_time:163359ms step_avg:100.53ms
step:1626/1770 train_time:163385ms step_avg:100.48ms
step:1627/1770 train_time:163478ms step_avg:100.48ms
step:1628/1770 train_time:163584ms step_avg:100.48ms
step:1629/1770 train_time:163689ms step_avg:100.48ms
step:1630/1770 train_time:163793ms step_avg:100.49ms
step:1631/1770 train_time:163896ms step_avg:100.49ms
step:1632/1770 train_time:163999ms step_avg:100.49ms
step:1633/1770 train_time:164105ms step_avg:100.49ms
step:1634/1770 train_time:164209ms step_avg:100.49ms
step:1635/1770 train_time:164314ms step_avg:100.50ms
step:1636/1770 train_time:164421ms step_avg:100.50ms
step:1637/1770 train_time:164529ms step_avg:100.51ms
step:1638/1770 train_time:164633ms step_avg:100.51ms
step:1639/1770 train_time:164738ms step_avg:100.51ms
step:1640/1770 train_time:164843ms step_avg:100.51ms
step:1641/1770 train_time:164948ms step_avg:100.52ms
step:1642/1770 train_time:165052ms step_avg:100.52ms
step:1643/1770 train_time:165156ms step_avg:100.52ms
step:1644/1770 train_time:165262ms step_avg:100.52ms
step:1645/1770 train_time:165367ms step_avg:100.53ms
step:1646/1770 train_time:165474ms step_avg:100.53ms
step:1647/1770 train_time:165580ms step_avg:100.53ms
step:1648/1770 train_time:165685ms step_avg:100.54ms
step:1649/1770 train_time:165790ms step_avg:100.54ms
step:1650/1770 train_time:165894ms step_avg:100.54ms
step:1651/1770 train_time:165998ms step_avg:100.54ms
step:1652/1770 train_time:166103ms step_avg:100.55ms
step:1653/1770 train_time:166208ms step_avg:100.55ms
step:1654/1770 train_time:166315ms step_avg:100.55ms
step:1655/1770 train_time:166422ms step_avg:100.56ms
step:1656/1770 train_time:166527ms step_avg:100.56ms
step:1657/1770 train_time:166632ms step_avg:100.56ms
step:1658/1770 train_time:166736ms step_avg:100.56ms
step:1659/1770 train_time:166842ms step_avg:100.57ms
step:1660/1770 train_time:166948ms step_avg:100.57ms
step:1661/1770 train_time:167053ms step_avg:100.57ms
step:1662/1770 train_time:167157ms step_avg:100.58ms
step:1663/1770 train_time:167261ms step_avg:100.58ms
step:1664/1770 train_time:167368ms step_avg:100.58ms
step:1665/1770 train_time:167473ms step_avg:100.58ms
step:1666/1770 train_time:167578ms step_avg:100.59ms
step:1667/1770 train_time:167682ms step_avg:100.59ms
step:1668/1770 train_time:167788ms step_avg:100.59ms
step:1669/1770 train_time:167893ms step_avg:100.60ms
step:1670/1770 train_time:167998ms step_avg:100.60ms
step:1671/1770 train_time:168103ms step_avg:100.60ms
step:1672/1770 train_time:168208ms step_avg:100.60ms
step:1673/1770 train_time:168315ms step_avg:100.61ms
step:1674/1770 train_time:168421ms step_avg:100.61ms
step:1675/1770 train_time:168525ms step_avg:100.61ms
step:1676/1770 train_time:168630ms step_avg:100.61ms
step:1677/1770 train_time:168738ms step_avg:100.62ms
step:1678/1770 train_time:168842ms step_avg:100.62ms
step:1679/1770 train_time:168947ms step_avg:100.62ms
step:1680/1770 train_time:169052ms step_avg:100.63ms
step:1681/1770 train_time:169157ms step_avg:100.63ms
step:1682/1770 train_time:169264ms step_avg:100.63ms
step:1683/1770 train_time:169369ms step_avg:100.64ms
step:1684/1770 train_time:169474ms step_avg:100.64ms
step:1685/1770 train_time:169578ms step_avg:100.64ms
step:1686/1770 train_time:169684ms step_avg:100.64ms
step:1687/1770 train_time:169792ms step_avg:100.65ms
step:1688/1770 train_time:169897ms step_avg:100.65ms
step:1689/1770 train_time:170002ms step_avg:100.65ms
step:1690/1770 train_time:170106ms step_avg:100.65ms
step:1691/1770 train_time:170211ms step_avg:100.66ms
step:1692/1770 train_time:170317ms step_avg:100.66ms
step:1693/1770 train_time:170423ms step_avg:100.66ms
step:1694/1770 train_time:170529ms step_avg:100.67ms
step:1695/1770 train_time:170634ms step_avg:100.67ms
step:1696/1770 train_time:170739ms step_avg:100.67ms
step:1697/1770 train_time:170846ms step_avg:100.68ms
step:1698/1770 train_time:170950ms step_avg:100.68ms
step:1699/1770 train_time:171054ms step_avg:100.68ms
step:1700/1770 train_time:171158ms step_avg:100.68ms
step:1701/1770 train_time:171263ms step_avg:100.68ms
step:1702/1770 train_time:171368ms step_avg:100.69ms
step:1703/1770 train_time:171473ms step_avg:100.69ms
step:1704/1770 train_time:171579ms step_avg:100.69ms
step:1705/1770 train_time:171684ms step_avg:100.69ms
step:1706/1770 train_time:171789ms step_avg:100.70ms
step:1707/1770 train_time:171897ms step_avg:100.70ms
step:1708/1770 train_time:172002ms step_avg:100.70ms
step:1709/1770 train_time:172109ms step_avg:100.71ms
step:1710/1770 train_time:172219ms step_avg:100.71ms
step:1711/1770 train_time:172330ms step_avg:100.72ms
step:1712/1770 train_time:172435ms step_avg:100.72ms
step:1713/1770 train_time:172540ms step_avg:100.72ms
step:1714/1770 train_time:172645ms step_avg:100.73ms
step:1715/1770 train_time:172753ms step_avg:100.73ms
step:1716/1770 train_time:172859ms step_avg:100.73ms
step:1717/1770 train_time:172964ms step_avg:100.74ms
step:1718/1770 train_time:173070ms step_avg:100.74ms
step:1719/1770 train_time:173176ms step_avg:100.74ms
step:1720/1770 train_time:173285ms step_avg:100.75ms
step:1721/1770 train_time:173392ms step_avg:100.75ms
step:1722/1770 train_time:173500ms step_avg:100.76ms
step:1723/1770 train_time:173606ms step_avg:100.76ms
step:1724/1770 train_time:173715ms step_avg:100.76ms
step:1725/1770 train_time:173823ms step_avg:100.77ms
step:1726/1770 train_time:173928ms step_avg:100.77ms
step:1727/1770 train_time:174036ms step_avg:100.77ms
step:1728/1770 train_time:174144ms step_avg:100.78ms
step:1729/1770 train_time:174250ms step_avg:100.78ms
step:1730/1770 train_time:174356ms step_avg:100.78ms
step:1731/1770 train_time:174465ms step_avg:100.79ms
step:1732/1770 train_time:174571ms step_avg:100.79ms
step:1733/1770 train_time:174679ms step_avg:100.80ms
step:1734/1770 train_time:174783ms step_avg:100.80ms
step:1735/1770 train_time:174889ms step_avg:100.80ms
step:1736/1770 train_time:174995ms step_avg:100.80ms
step:1737/1770 train_time:175100ms step_avg:100.81ms
step:1738/1770 train_time:175207ms step_avg:100.81ms
step:1739/1770 train_time:175313ms step_avg:100.81ms
step:1740/1770 train_time:175419ms step_avg:100.82ms
step:1741/1770 train_time:175525ms step_avg:100.82ms
step:1742/1770 train_time:175636ms step_avg:100.82ms
step:1743/1770 train_time:175741ms step_avg:100.83ms
step:1744/1770 train_time:175847ms step_avg:100.83ms
step:1745/1770 train_time:175955ms step_avg:100.83ms
step:1746/1770 train_time:176064ms step_avg:100.84ms
step:1747/1770 train_time:176168ms step_avg:100.84ms
step:1748/1770 train_time:176277ms step_avg:100.84ms
step:1749/1770 train_time:176382ms step_avg:100.85ms
step:1750/1770 train_time:176488ms step_avg:100.85ms
step:1750/1770 val_loss:3.2812 train_time:176592ms step_avg:100.91ms
step:1751/1770 train_time:176619ms step_avg:100.87ms
step:1752/1770 train_time:176710ms step_avg:100.86ms
step:1753/1770 train_time:176814ms step_avg:100.86ms
step:1754/1770 train_time:176920ms step_avg:100.87ms
step:1755/1770 train_time:177026ms step_avg:100.87ms
step:1756/1770 train_time:177132ms step_avg:100.87ms
step:1757/1770 train_time:177238ms step_avg:100.88ms
step:1758/1770 train_time:177343ms step_avg:100.88ms
step:1759/1770 train_time:177450ms step_avg:100.88ms
step:1760/1770 train_time:177555ms step_avg:100.88ms
step:1761/1770 train_time:177662ms step_avg:100.89ms
step:1762/1770 train_time:177772ms step_avg:100.89ms
step:1763/1770 train_time:177876ms step_avg:100.89ms
step:1764/1770 train_time:177982ms step_avg:100.90ms
step:1765/1770 train_time:178088ms step_avg:100.90ms
step:1766/1770 train_time:178197ms step_avg:100.90ms
step:1767/1770 train_time:178301ms step_avg:100.91ms
step:1768/1770 train_time:178407ms step_avg:100.91ms
step:1769/1770 train_time:178512ms step_avg:100.91ms
step:1770/1770 train_time:178618ms step_avg:100.91ms
step:1770/1770 val_loss:3.2782 train_time:178723ms step_avg:100.97ms
peak memory allocated: 30724 MiB reserved: 46472 MiB
