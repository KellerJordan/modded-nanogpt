import os
import sys

with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import contextlib
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
import torch._inductor.config as config
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# Use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.nn.parallel import DistributedDataParallel as DDP

config.coordinate_descent_tuning = True

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
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: "list[Tensor]" = [*params]
        assert all(isinstance(p, Tensor) for p in params)
        sizes = {p.numel() for p in params}
        param_groups = [dict(
            params=[p for p in params if p.numel() == size],
            update_buffer=[torch.empty(size, device='cuda', dtype=torch.bfloat16) for _ in range(world_size)],
        ) for size in sizes]
        super().__init__(param_groups, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            update_buffers: "list[Tensor]" = group['update_buffer']
            # generate weight updates in distributed fashion
            params: "list[Tensor]" = group['params']
            handle = None
            params_world = None
            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffers):
                    p_world.data.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf: Tensor = state['momentum_buffer']
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffers[rank]
                update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather(update_buffers, g, async_op=True)
                params_world = params[base_i : base_i + world_size]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)
    
    def reset_parameters(self) -> None:
        # leave for future tuning
        # 0.5 is a bit better than the default 1/sqrt(3)
        std = 0.5 * (self.in_features ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=65536):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0.0, 1.0, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i, j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x: Tensor):
        cos, sin = self.cos[None, :x.size(-3), None, :], self.sin[None, :x.size(-3), None, :]
        x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_heads) # dim // num_heads = head_dim
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, 'Must use batch size = 1 for FlexAttention'
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        if ve is None: # skip mid-layers token value embeddings by @YouJiacheng 
            v = self.lambdas[0] * v
        else:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(model_dim, num_heads) if layer_idx != 7 else None
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.layer_idx = layer_idx

    def forward(self, x, vi, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), vi, block_mask)
        x = x + self.mlp(norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.__setattr__
        self.embed = nn.ModuleList([nn.Embedding(num_embeddings, embedding_dim) for _ in range(3)])

    def forward(self, inputs) -> "list[Tensor | None]":
        ve = [emb(inputs) for emb in self.embed]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved upon @leloykun's U-net structure
        ve = [
            ve[0], ve[1], ve[2], None, None, None,
            None, None, None, ve[0], ve[1], ve[2],
        ]
        return ve


# -----------------------------------------------------------------------------
# The main GPT-2 model

def next_multiple_of_128(v: float | int):
    n = 128
    return next(x for x in range(int(v) + 1 + n)[::n] if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
        self.value_embeds = ValueEmbedding(vocab_size, model_dim)

        self.blocks = nn.ModuleList([Block(model_dim, num_heads, layer_idx) for layer_idx in range(num_layers)])
        
        # U-net design by @brendanh0gan
        self.num_encoder_layers = num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_128(vocab_size))
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        sliding_window_num_blocks: Tensor,
    ):
        BLOCK_SIZE = 128
        assert inputs.ndim == 1
        assert len(inputs) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(inputs) // BLOCK_SIZE
        docs = (inputs == 50256).cumsum(0)
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask
        
        def dense_to_ordered(dense_mask: Tensor):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        def create_doc_swc_block_mask(sliding_window_num_blocks: Tensor):
            kv_idx = block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
            q_idx = block_idx[:, None]
            causal_bm = q_idx >= kv_idx
            causal_full_bm = q_idx > kv_idx
            window_bm = q_idx - kv_idx < sliding_window_num_blocks
            window_full_bm = window_bm # block-wise sliding window by @YouJiacheng
            # document_bm = (docs_low[q_idx] <= docs_high[kv_idx]) & (docs_low[kv_idx] <= docs_high[q_idx])
            document_bm = (docs_low[:, None] <= docs_high) & (docs_low <= docs_high[:, None])
            document_full_bm = (docs_low[:, None] == docs_high) & (docs_low == docs_high[:, None])
            nonzero_bm = causal_bm & window_bm & document_bm
            full_bm  = causal_full_bm & window_full_bm & document_full_bm
            kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm & ~full_bm)
            full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)
            return BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        block_mask = create_doc_swc_block_mask(sliding_window_num_blocks)

        # forward the GPT model itself
        x = self.embed(inputs)[None]
        x = norm(x) # @Grad62304977
        x0 = x
        ve = self.value_embeds(inputs)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_mask)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            # U-net structure on token value embeddings by @leloykun
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
    assert header[1] == 1, 'unsupported version'
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open('rb', buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, 'number of tokens read does not match header?'
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern: str, batch_size: int):
        assert batch_size % world_size == 0
        self.files = sorted(Path.cwd().glob(filename_pattern))
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.next_shard = 0
        self.advance()

    def advance(self): # advance to next data shard
        self.pos = 0
        self.tokens = _load_data_shard(self.files[self.next_shard])
        self.next_shard = (self.next_shard + 1) % len(self.files)

    def next_batch(self):
        local_batch_size = self.batch_size // world_size
        buf = self.tokens[self.pos + rank * local_batch_size:][:local_batch_size + 1]
        # by @YouJiacheng: host side async is sufficient;
        # no performance improvement was observed when introducing a separate stream.
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # inputs
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # targets
        # advance current position and load next shard if necessary
        self.pos += self.batch_size
        if self.pos + self.batch_size + 1 >= len(self.tokens):
            self.advance()
        return inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_bin = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    val_bin = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization
    batch_size = 8*64*1024 # batch size in tokens
    num_iterations = 1490 # number of iterations to run
    cooldown_iters = 600 # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # implementation
    save_checkpoint = False
args = Hyperparameters()

# set up DDP (distributed data parallel). torchrun sets this env variable
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
assert torch.cuda.is_available()
device = torch.device('cuda', local_rank)
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl', device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
def print0(s, console=False): ...
if master_process:
    run_id = uuid.uuid4()
    (logs_dir := Path("logs")).mkdir(exist_ok=True)
    logfile = logs_dir / f"{run_id}.txt"
    print(logfile.stem)
    def print0(s, console=False):
        with logfile.open("a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0('='*100)
# log information about the hardware/software environment this is running on
print0(f'Running Python {sys.version}')
print0(f'Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}')
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0('='*100)

# load data
train_loader = DistributedDataLoader(args.train_bin, args.batch_size)
val_loader = DistributedDataLoader(args.val_bin, args.batch_size)
print0(f'Training dataloader files: {train_loader.files}')
print0(f'Validation dataloader files: {val_loader.files}')
print0('='*100)
inputs_train, targets_train = train_loader.next_batch()

model = GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768)
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
model: nn.Module = torch.compile(model)
ddp_model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
sliding_window_num_blocks = torch.tensor(1, dtype=torch.int32, device="cuda")
sw_num_blocks_prev = 1

# collect the parameters to optimize
hidden_matrix_params = [p for p in model.blocks.parameters() if p.ndim == 2]
embed_params = [model.embed.weight, *model.value_embeds.parameters()]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
optimizer1 = torch.optim.Adam([
        dict(params=head_params, lr=0.008),
        dict(params=embed_params, lr=0.6),
        dict(params=scalar_params, lr=0.04),
    ],
    betas=(0.8, 0.95),
    fused=True,
)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95)
optimizers = [optimizer1, optimizer2]

# learning rate decay scheduler (stable then decay)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) constant lr for first part of training
    if it < args.num_iterations - args.cooldown_iters:
        return 1.0
    # 2) then linear cooldown
    else:
        return (args.num_iterations - it) / args.cooldown_iters
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]


# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # Linearly increase the sliding window size over training in chunks of 128 from 128 -> 1792. By @fernbear.bsky.social
    frac_done = step / train_steps # training progress
    sw_num_blocks = next_multiple_of_128(max(1, 1728 * frac_done)) // 128
    if sw_num_blocks != sw_num_blocks_prev:
        sliding_window_num_blocks.copy_(sw_num_blocks, non_blocking=True)
        sw_num_blocks_prev = sw_num_blocks

    # --------------- VALIDATION SECTION -----------------
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        # run validation batches
        ddp_model.eval()
        val_loader.reset()
        val_loss = 0.0
        # calculate the number of steps to take in the val loop.
        assert args.val_tokens % args.batch_size == 0
        val_steps = args.val_tokens // args.batch_size
        for _ in range(val_steps):
            with torch.no_grad():
                inputs_val, targets_val = val_loader.next_batch()
                val_loss += ddp_model(inputs_val, targets_val, sliding_window_num_blocks)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # logging
        print0(f'step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms', console=True)
        ddp_model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f'logs/{run_id}', exist_ok=True)
            torch.save(log, f'logs/{run_id}/state_step{step:06d}.pt')
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    with contextlib.ExitStack() as stack:
        if step >= 5:
            stack.enter_context(torch.compiler.set_stance(skip_guard_eval_unsafe=True))
        loss = ddp_model(inputs_train, targets_train, sliding_window_num_blocks)
        loss.backward()
        del loss
        inputs_train, targets_train = train_loader.next_batch()
    # momentum warmup for Muon
    frac = min(step/300, 1)
    for group in optimizer2.param_groups:
        group['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    ddp_model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f'step:{step+1}/{train_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms', console=True)

print0(
    f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
)
dist.destroy_process_group()
