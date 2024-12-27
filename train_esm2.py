import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from transformers import EsmTokenizer
from dataclasses import dataclass
from pathlib import Path
from utils import ProteinMasker


# -----------------------------------------------------------------------------
# Muon optimizer
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
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
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(isinstance(p, torch.Tensor) for p in params)
        sizes = {p.numel() for p in params}
        param_groups = [
            {
                'params': [p for p in params if p.numel() == size],
                'update_buffer': [
                    torch.empty(size, device='cuda', dtype=torch.bfloat16)
                    for _ in range(self.world_size)
                ],
            }
            for size in sizes
        ]
        super().__init__(param_groups, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            update_buffers = group['update_buffer']
            # generate weight updates in distributed fashion
            params = group['params']
            assert len(params) % self.world_size == 0
            handle = None
            params_world = None
            def update_prev():
                if params_world is None:
                    return
                if handle is not None:
                    handle.wait()
                for p_world, g_world in zip(params_world, update_buffers):
                    p_world.data.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.world_size]:
                p = params[base_i + self.rank]
                g = p.grad
                assert g is not None
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.lerp_(g, 1 - momentum)
                g = g.lerp_(buf, momentum) if nesterov else buf
                g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                update_prev()
                if self.world_size > 1:
                    handle = dist.all_gather(update_buffers, g, async_op=True)
                else:
                    update_buffers[0].copy_(g)
                    handle = None
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions
def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
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

    def forward(self, x, vi, block_mask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        v = self.lambdas[0] * v + self.lambdas[1] * vi.view_as(v) # @KoszarskyB & @Grad62304977
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, enable_gqa=True)
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
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config.model_dim, config.num_heads)
        self.mlp = MLP(config.model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, vi, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), vi, block_mask)
        x = x + self.mlp(norm(x))
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.model_dim)
            for _ in range(6)
        ])

    def forward(self, inputs) -> "list[torch.Tensor]":
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve


# -----------------------------------------------------------------------------
# The main ESM Bert model
class ESM(nn.Module):
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.masker = ProteinMasker(tokenizer, 0.20)
        self.cls_id = tokenizer.cls_token_id
        self.vocab_size = tokenizer.vocab_size
        self.num_layers = config.num_layers

        # U-net design by @brendanh0gan
        assert config.num_layers % 2 == 0, "Number of layers should be even for U-net design"
        self.num_encoder_layers = config.num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.embed = nn.Embedding(self.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
        # U-net structure on token value embeddings by @leloykun
        self.value_embeds = ValueEmbedding(config)
        self.lm_head = CastedLinear(config.model_dim, self.vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977
        self.cross_entropy = nn.CrossEntropyLoss()

    def get_logits(self, input_seq: torch.Tensor, sliding_window_size: torch.Tensor):
        docs = (input_seq == self.cls_id).cumsum(0)

        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
            doc_mask = docs[q_idx] == docs[kv_idx]
            return bidirectional_sliding_window_mask & doc_mask

        S = len(input_seq)
        block_mask = create_block_mask(doc_mask_mod, None, None, S, S)

        x = self.embed(input_seq[None])
        x = norm(x) # @Grad62304977
        x0 = x
        ve = self.value_embeds(input_seq)
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
        return logits

    def inference(self, input_ids, labels=None): # premaked input_ids
        sliding_window_size = torch.tensor(2048, dtype=torch.int32, device="cuda")
        logits = self.get_logits(input_ids, sliding_window_size)
        loss = None
        if labels is not None:
            loss = self.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
        return loss, logits

    def forward(self, input_ids, sliding_window_size: torch.Tensor):
        input_ids, labels = self.masker(input_ids)
        logits = self.get_logits(input_ids, sliding_window_size)
        return self.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader
def _peek_data_shard(file: Path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    return int(header[2]) # number of tokens (claimed)


def _load_data_shard(path: Path, num_tokens):
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint8, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == num_tokens, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, seq_len, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.seq_len = seq_len

        # glob files that match the pattern
        self.files = sorted(Path.cwd().glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        self.files_num_tokens = [_peek_data_shard(file) for file in self.files]
        assert min(self.files_num_tokens) >= num_processes * seq_len + 1
        self.total_num_tokens = sum(self.files_num_tokens)

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.seq_len
        self.tokens = _load_data_shard(self.files[self.current_shard], self.files_num_tokens[self.current_shard])

    def next_batch(self):
        batch_size = self.seq_len * self.num_processes
        buf = self.tokens[self.current_position:self.current_position+self.seq_len+1]
        # host side async is sufficient;
        # no performance improvement was observed when introducing a separate stream.
        seq = buf.to(device="cuda", dtype=torch.int32, non_blocking=True) # inputs
        # advance current position and load next shard if necessary
        self.current_position += batch_size
        if self.current_position + batch_size + 1 >= len(self.tokens):
            self.advance()
        return seq
    

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        from huggingface_hub import hf_hub_download 
        from datasets import Dataset as HFDataset
        local_file = hf_hub_download(
            repo_id="Synthyra/omg_prot50",
            filename=f"data/test-00000-of-00001.parquet",
            repo_type="dataset"
        )
        data = HFDataset.from_parquet(local_file)
        print(data)
        sequences = data['sequence']
        self.sequences = sorted(sequences, key=len, reverse=True)
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.masker = ProteinMasker(tokenizer, 0.15)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids = self.tokenizer(self.sequences[idx], return_tensors='pt', truncation=True, max_length=1024).input_ids
        input_ids, labels = self.masker(input_ids)
        return input_ids, labels
        

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return input_ids, labels


# -----------------------------------------------------------------------------
# int main
@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/omgprot50/omgprot50_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/omgprot50/omgprot50_valid_*.bin'  # input .bin to eval validation loss on
    #input_test_bin : str = 'data/omgprot50/omgprot50_test_*.bin'  # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8 # batch size, in sequences, across all devices
    sequence_length : int = 64*1024 # sequence length, in tokens
    num_iterations : int = 1480 # number of iterations to run
    warmup_iters : int = 0
    cooldown_iters : int = 600 # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every = None # save every how many steps? None for no saving


@dataclass
class ModelConfig:
    # 33 tokens: https://huggingface.co/Synthyra/ESMplusplus_large/blob/main/modeling_esm_plusplus.py#L868-L874
    # Depth of the number of layers is typically more important than the depth of the hidden dimension for PLMs
    # ESM2-8M has 6 layers, 20 heads, 320 hidden dim: https://huggingface.co/facebook/esm2_t6_8M_UR50D/blob/main/config.json
    # ESM2-35M has 12 layers, 20 heads, 480 hidden dim: https://huggingface.co/facebook/esm2_t12_35M_UR50D/blob/main/config.json
    # ESM2-150M has 30 layers, 20 heads, 640 hidden dim: https://huggingface.co/facebook/esm2_t30_150M_UR50D/blob/main/config.json
    # ESM2-650M has 33 layers, 20 heads, 1280 hidden dim: https://huggingface.co/facebook/esm2_t33_650M_UR50D/blob/main/config.json
    num_layers : int = 12
    num_heads : int = 6 # head dim 128 suggested by @Grad62304977
    model_dim : int = 768


model_config = ModelConfig()
args = Hyperparameters()


def get_param_count(model):
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
    return total_params

# set up DDP (distributed data parallel) if available, otherwise single GPU
if 'RANK' in os.environ:
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{ddp_local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', device_id=device)
    dist.barrier()
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0 
    ddp_world_size = 1
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    master_process = True

print(f'using device: {device}')

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    Path('logs').mkdir(exist_ok=True)
    # logdir = Path('logs') / f'{run_id}'
    # logdir.mkdir()
    logfile = Path('logs') / f'{run_id}.txt'
    print(logfile.stem)
    # create the log file
    with logfile.open('w') as f:
        # begin the log by printing this file (the Python code)
        print(code, file=f)
        print('=' * 100, file=f)
def print0(s, logonly=False):
    if master_process:
        with logfile.open('a') as f:
            if not logonly:
                print(s)
            print(s, file=f)
# log information about the hardware/software environment this is running on
# and print the full `nvidia-smi` to file
print0(f'Running python {sys.version}')
print0(f'Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:')
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('='*100, logonly=True)

# calculate the number of steps to take in the val loop.
assert args.val_tokens % (args.sequence_length * ddp_world_size) == 0
val_steps = args.val_tokens // (args.sequence_length * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (ddp_world_size) == 0
train_accumulation_steps = args.batch_size // ddp_world_size

# load tokens
train_loader = DistributedDataLoader(args.input_bin, args.sequence_length, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, args.sequence_length, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: total number of tokens: {train_loader.total_num_tokens} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.total_num_tokens} across {len(val_loader.files)} files")
print0('='*100, logonly=True)
seq_train = train_loader.next_batch()

model = ESM(model_config)
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# wrap model in DDP only if using distributed training
if ddp_world_size > 1:
    model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
    raw_model = model.module
else:
    raw_model = model

# init the optimizer(s)
embed_params = [*raw_model.embed.parameters(), *raw_model.value_embeds.parameters()]
optimizer1 = torch.optim.Adam(embed_params, lr=0.6, betas=(0.8, 0.95), fused=True)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.008, betas=(0.8, 0.95), fused=True)
params = list(raw_model.blocks.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
optimizer3 = Muon(matrix_params, lr=0.05, momentum=0.95)
optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True)
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

# learning rate decay scheduler (linear warmup and cooldown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    # 3) linear cooldown
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

sliding_window_size = torch.tensor(1024 - 128, dtype=torch.int32, device="cuda")
sw_prev = 1024 - 128
# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()

### BEGIN TRAINING LOOP ###
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # Linearly increase the sliding window size over training in chunks of 128 from 1024 -> 2048. By @fernbear.bsky.social
    frac_done = step / args.num_iterations # training progress
    sw_size = int(((1 - frac_done) * 1023 + frac_done * 2048) // 128) * 128
    if sw_size != sw_prev:
        sliding_window_size.copy_(sw_size, non_blocking=True)
        sw_prev = sw_size

    # once in a while evaluate the validation dataset
    if args.val_loss_every > 0 and step % args.val_loss_every == 0:
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(val_steps):
                seq_val = val_loader.next_batch()
                val_loss += model(seq_val, sliding_window_size)
        if ddp_world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms perplexity:{(math.e**val_loss):.4f} param_count:{get_param_count(model):,}')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    # save checkpoint every `save_every` steps
    if master_process and args.save_every:
        if last_step or (step % args.save_every == 0):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            # save the state of the training process
            log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

    if last_step:
        break

    # --------------- FORWARD AND BACKWARD PASS -----------------
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        with contextlib.ExitStack() as stack:
            if ddp_world_size > 1 and i < train_accumulation_steps: # there's no need to sync gradients every accumulation step
                stack.enter_context(model.no_sync())
            #if step >= 5:
            #    stack.enter_context(torch.compiler.set_stance(skip_guard_eval_unsafe=True))
            model(seq_train, sliding_window_size).backward()
            seq_train = train_loader.next_batch()
    if train_accumulation_steps != 1:
        for p in model.parameters():
            p.grad /= train_accumulation_steps
    # momentum warmup for Muon
    frac = min(step/300, 1)
    for group in optimizer3.param_groups:
        group['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- FORWARD AND BACKWARD PASS END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

# Finish timing before inference
torch.cuda.synchronize()
torch.manual_seed(42)
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
model.eval()
results, all_true, all_pred = [], [], []
total_loss, count = 0.0, 0
test_loader = DataLoader(TestDataset(), batch_size=args.batch_size, collate_fn=collate_fn)

with torch.no_grad():
    for input_ids, labels in test_loader:
        loss, logits = model.inference(input_ids, labels)        
        total_loss += loss.item()
        count += 1
        all_true.extend(labels.cpu().numpy().flatten())
        all_pred.extend(logits.argmax(dim=-1).cpu().numpy().flatten())

average_loss = total_loss / count
perplexity = torch.exp(torch.tensor(average_loss)).item()
all_true = np.array(all_true)
all_pred = np.array(all_pred)
mask = (all_true != -100)
all_true = all_true[mask]
all_pred = all_pred[mask]

precision = precision_score(all_true, all_pred, average='weighted')
recall = recall_score(all_true, all_pred, average='weighted')
f1 = f1_score(all_true, all_pred, average='weighted')
accuracy = accuracy_score(all_true, all_pred)
mcc = matthews_corrcoef(all_true, all_pred)

print0("Final Results:")
print0(f"  Loss:        {average_loss:.4f}")
print0(f"  Perplexity:  {perplexity:.4f}")
print0(f"  Precision:   {precision:.4f}")
print0(f"  Recall:      {recall:.4f}")
print0(f"  F1:          {f1:.4f}")
print0(f"  Accuracy:    {accuracy:.4f}")
print0(f"  MCC:         {mcc:.4f}")

print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
if ddp_world_size > 1:
    dist.destroy_process_group()
