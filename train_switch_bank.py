# ========== train_switch_bank.py ==========
# Based on the "speedrun" baseline".

import csv
import os
import sys
from pathlib import Path

def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""

code_paths = [
    Path(sys.argv[0]).resolve(),
    Path("switch_bank/utils.py"),
    Path("switch_bank/optim/muon.py"),
    Path("switch_bank/model/components.py"),
    Path("switch_bank/model/gpt.py"),
    Path("switch_bank/data.py"),
    Path("switch_bank/trainer.py"),
]
code_parts = []
for p in code_paths:
    if p.exists():
        code_parts.append(f"===== {p} =====\\n{_read_text(p)}")
code = "\\n\\n".join(code_parts)
import uuid
import time
import copy
from dataclasses import dataclass
from switch_bank.utils import compute_train_micro_len
from switch_bank.optim.muon import Muon
from switch_bank.model.components import CausalSelfAttention
from switch_bank.model.gpt import GPT
from switch_bank.data import summarize_router_metrics, summarize_expert_usage, summarize_expert_activity, router_summary_str
from switch_bank import trainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch._functorch.config
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
torch._functorch.config.donated_buffer = False

# ***** SET ME TRUE FOR NVIDIA? ******
torch._dynamo.config.compiled_autograd = False


#
# -----------------------------------------------------------------------------
# The main model
# -----------------------------------------------------------------------------

# ----- Parameter accounting / logging -----
def _num_params(tensors_iter):
    return sum(p.numel() for p in tensors_iter)

def _fmt(n: int) -> str:
    return f"{n:,} ({n/1e6:.3f}M)"

@torch.no_grad()
def log_param_counts(model: nn.Module):
    #if not args.enable_extra_logging:
    #    return
    # totals
    total = _num_params(model.parameters())

    # attention stack: merged QKV/out per block that has attention
    attn_params = []
    attn_layers = 0
    for b in model.blocks:
        if isinstance(b.attn, CausalSelfAttention):
            attn_params.append(b.attn.qkvo_w)
            attn_layers += 1
    attn_total = _num_params(attn_params)

    # FFN bank: experts + routers
    bank_expert_params = list(model.bank.W1) + list(model.bank.W2)
    bank_router_params = list(model.bank.router_w) + list(model.bank.router_b)
    bank_expert_total = _num_params(bank_expert_params)
    bank_router_total = _num_params(bank_router_params)
    bank_total = bank_expert_total + bank_router_total

    # embeddings: tied token embedding + N value-embedding tables
    tok_embed_total = _num_params(model.embed.parameters())
    ve_total = sum(_num_params(ve.parameters()) for ve in model.value_embeds)
    embeds_total = tok_embed_total + ve_total

    # lm head (if untied / instantiated)
    head_total = _num_params([model.lm_head]) if model.lm_head is not None else 0

    # scalars (skip lambdas / SA lambdas / skip weights)
    scalars_total = model.scalars.numel()

    adapter_total = 0
    if getattr(model.bank, "use_adapters", False):
        adapter_total = model.bank.adapter_scale.numel() + model.bank.adapter_bias.numel()

    # anything unaccounted (should be ~0; keeps us honest)
    accounted = attn_total + bank_total + embeds_total + head_total + scalars_total + adapter_total
    other_total = total - accounted

    # pretty print
    print0("=== Parameter counts ===", console=True)
    print0(f"model total:           {_fmt(total)}", console=True)
    print0(f"  attention stack ({attn_layers} of {args.num_layers} layers run attention): {_fmt(attn_total)}", console=True)
    print0(f"  FFN bank total:      {_fmt(bank_total)}", console=True)
    print0(f"    ├─ experts W1/W2:  {_fmt(bank_expert_total)}", console=True)
    print0(f"    └─ routers:        {_fmt(bank_router_total)}", console=True)
    print0(f"  embeddings (tok + {model.num_value_embeds}× value): {_fmt(embeds_total)}", console=True)
    print0(f"    └─ token embed:    {_fmt(tok_embed_total)}", console=True)
    print0(f"    └─ value embeds:   {_fmt(ve_total)}", console=True)
    if head_total:
        tied_state = "tied" if model._head_tied_runtime else "untied"
        print0(f"  lm head ({tied_state}):   {_fmt(head_total)}", console=True)
    if adapter_total:
        print0(f"  adapters:            {_fmt(adapter_total)}", console=True)
    print0(f"  scalars:             {_fmt(scalars_total)}", console=True)
    if other_total != 0:
        print0(f"  other (unclassified): {_fmt(other_total)}", console=True)
    print0("="*100, console=False)

# -----------------------------------------------------------------------------
# int main
# -----------------------------------------------------------------------------

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin"
    val_files = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens = 32768 * 20 #10485760
    train_seq_len = 48*1024 #64*1024          # effective tokens per optimizer step per rank
    val_seq_len = 8192 #4*64*1024
    # minibatch / gradient accumulation
    grad_accum_steps = 12 # 8 for 32k, 12 for 48k, 16 for 64k    # default=1 keeps original behavior
    train_micro_seq_len: int | None = None  # if None, computed as train_seq_len // grad_accum_steps
    # optimization
    num_iterations = 4123 #2980 #5960
    cooldown_frac = 0.55 #0.7
    lr_final_mult = 0.0  # decay to this % of original lr at final iteration
    lr_freeze_last_steps = 150  # decay toward lr_final_mult at final step, but freeze lr at num_iterations-lr_freeze_last_steps
    lr_embed = 0.3
    lr_scalar = 0.015
    lr_head = 1/320
    lr_router = 0.095
    lr_adapter = 0.03
    lr_muon = 0.025
    router_grad_clip_norm = 0.0 #1.0
    router_autoclip = True #False
    # architecture
    vocab_size = 50257
    model_dim = 896
    num_layers = 16
    head_dim = 128
    num_heads = model_dim // head_dim #7
    # value-embeddings integer count: 0, 1, 2, or 3 supported.
    num_value_embeds = 2
    tie_lm_head = False
    untie_lm_head_frac = -1.0
    # Bank / routing
    num_experts = 9
    ffn_hidden = 1024
    topk = 1
    topk_val: int | None = None
    lb_coeff = 2.15e-3
    router_entropy_coeff = 2.5e-3  # coefficient for router entropy aux loss component
    use_router_adapters = True
    router_block_pos_bins = 8  # 4 / 8 / 16
    first_doc_tokens_N = 64
    router_enable_forward_ema = False
    router_enable_reverse_ema = True
    ema_alpha_fwd = 0.80
    ema_alpha_rev = 0.85
    ema_window_size_fwd = 128  # <=0 means full sequence
    ema_block_size_fwd = 128
    ema_window_size_rev = 384
    ema_block_size_rev = 384
    router_ema_layer_stride = -1  # How often to calculate fresh EMAs (which are then used by the next N-1 layers).  N < 0 -> num_layers (one shared EMA calculation for all layers).
    # Parameter freezing
    router_freeze_frac = 1.0
    router_freeze_adapters = False
    router_lr_reduce_start_frac = -1.0
    shared_ffn_freeze_frac = 1.0
    shared_ffn_lr_reduce_start_frac = -1.0
    # skip-attention layers (short-SWA) — exactly two
    skip_attn_layers = (7, )
    expert_activation_schedule: tuple[tuple[int, int], ...] = ((0, 1), (200, 2), (375, 3), (625, 4), (900, 5), (1175, 6), (1575, 7), (1850, 8), (2175, 9))
    router_temp_init = 1.85
    router_temp_final = 0.65
    router_temp_power = 1.5  # fallback if anchor disabled
    router_temp_anchor_delta_steps = 350  # steps after 2nd expert activation to hit anchor ratio
    router_temp_anchor_ratio = 0.49  # temp curve hits this ratio at anchor delta
    router_logit_cap_initial = 1.0
    router_logit_cap_final = 20.0
    router_logit_cap_delta_steps = 390 # ramp length after second expert activation
    # Optional Gumbel exploration (off by default)
    router_use_gumbel = True
    router_gumbel_schedule: tuple[tuple[int, int], ...] =  ((200, 1175), (1225, 1300), (1425, 1900), (2400, 2425), (2725, 2750), (2925, 2950), (3200, 3225), (3475, 3500), (3925, -1))  # ensure ~2500 active before end
    # Layerwise router temp & lb boosts.
    router_boost_shape = "peak"  # options: peak (default), valley, linear_start, linear_end
    router_temp_boost = 0.2
    router_lb_boost = 0.5
    router_layer_peak_frac = 0.475  # only used for peak or valley shapes. boosts are calculated continuously
    # evaluation and logging
    val_loss_every = 250 #125  # 0 for only at end
    save_final_checkpoint = False
    checkpoint_save_step: int = -1  # -1 disables mid-training save
    resume_checkpoint: str | None = None #"./logs/375/state_step000375.pt"
    use_wandb = True
    wandb_project = "switch-bank-long"
    wandb_run_name = ""
    wandb_log_every = 1
    enable_extra_logging = False
    enable_extra_wandb_logging = True
    do_model_warmup = False
    metrics_log_every = 25

args = Hyperparameters()
if args.router_ema_layer_stride < 0:
    args.router_ema_layer_stride = args.num_layers

def hyperparams_to_config(h: Hyperparameters) -> dict:
    cfg: dict[str, object] = {}
    for name in dir(h):
        if name.startswith("_"):
            continue
        value = getattr(h, name)
        if callable(value):
            continue
        cfg[name] = value
    return cfg

untie_lm_head_after = -1
if args.tie_lm_head and args.untie_lm_head_frac is not None and args.untie_lm_head_frac >= 0:
    untie_lm_head_after = int(args.untie_lm_head_frac * args.num_iterations)
    untie_lm_head_after = min(max(untie_lm_head_after, 0), args.num_iterations)

run_id = int(os.environ.get("RUN_ID", 0))
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
#assert world_size == 8
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0)
run_id_full: str | None = None

if master_process:
    run_id_full = f"{run_id:03d}_{uuid.uuid4()}"
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id_full}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# --- Robust Inductor trace hook (compatible with callsites with/without metadata_fn) ---
from torch._logging._internal import trace_structured as _orig_trace_structured  # keep original
import torch._inductor.codecache  # noqa: E402
import torch._inductor.graph      # noqa: E402

def _patched_trace_structured(name, *args, **kwargs):
    """
    Torch Inductor sometimes calls trace_structured(name, metadata_fn, **kwargs),
    and other times as trace_structured(name, **kwargs) with metadata_fn omitted.
    Be permissive and forward both forms. Also print compiled filename when available.
    """
    metadata_fn = kwargs.get("metadata_fn", None)
    if metadata_fn is None and len(args) > 0 and callable(args[0]):
        # first positional could be metadata_fn
        metadata_fn = args[0]
    try:
        if name == "inductor_output_code" and callable(metadata_fn):
            md = metadata_fn()
            filename = (md.get("filename", "Unknown") if isinstance(md, dict) else "Unknown")
            print0(f"inductor_output_code: {filename}")
    except Exception:
        # never let logging break compilation
        pass
    return _orig_trace_structured(name, *args, **kwargs)

torch._inductor.codecache.trace_structured = _patched_trace_structured
torch._inductor.graph.trace_structured = _patched_trace_structured
# --- end robust hook ---

wandb_run = None
if args.use_wandb and os.environ.get("WANDB_DISABLED", "0").lower() not in ("1", "true", "yes") and master_process:
    try:
        import wandb  # type: ignore
        wandb_run = wandb.init(
            project=args.wandb_project,
            config=hyperparams_to_config(args),
            #name=args.wandb_run_name or run_id_full or f"rank{rank}",
        )
        print0("wandb logging enabled.", console=True)
    except Exception as err:
        print0(f"wandb init failed ({err}); disabling wandb.", console=True)
        wandb_run = None

metrics_csv_file = None
metrics_csv_writer = None
expert_usage_headers: list[str] = []
expert_active_headers: list[str] = []
if master_process and run_id_full is not None and args.enable_extra_logging:
    metrics_csv_path = f"logs/{run_id_full}_metrics.csv"
    metrics_csv_file = open(metrics_csv_path, "w", newline="")
    metrics_csv_writer = csv.writer(metrics_csv_file)
    expert_usage_headers = [f"expert_usage_e{i}" for i in range(args.num_experts)]
    expert_active_headers = [f"expert_active_e{i}" for i in range(args.num_experts)]
    router_ema_headers: list[str] = []
    if args.router_enable_forward_ema:
        router_ema_headers.append("router_ema_alpha_forward")
    if args.router_enable_reverse_ema:
        router_ema_headers.append("router_ema_alpha_reverse")
    metrics_csv_writer.writerow([
        "step", "loss", "loss_main", "loss_aux",
        "router_imp_cv2", "router_load_cv2", "router_usage_frac",
        "router_topk_prob_mean", *router_ema_headers, "router_max_logit",
        "logit_cap", "router_temp", "window_blocks", *expert_usage_headers, *expert_active_headers
    ])

def log_metrics_row(step_value: int, avg_loss: float, avg_main: float, avg_aux: float,
                    router_summary: dict[str, float], logit_cap_value: float | None,
                    router_temp_value: float, window_blocks_value: int,
                    expert_usage: torch.Tensor | None,
                    expert_active: torch.Tensor | None):
    if metrics_csv_writer is None:
        return
    expert_usage_list = []
    if expert_usage is not None:
        expert_usage_list = [float(x) for x in expert_usage.tolist()]
    else:
        expert_usage_list = [float("nan")] * len(expert_usage_headers)
    expert_active_list = []
    if expert_active is not None:
        expert_active_list = [float(x) for x in expert_active.tolist()]
    else:
        expert_active_list = [float("nan")] * len(expert_active_headers)
    row = [
        step_value,
        avg_loss,
        avg_main,
        avg_aux,
        router_summary.get("imp_cv2", float("nan")),
        router_summary.get("load_cv2", float("nan")),
        router_summary.get("usage_frac", float("nan")),
        router_summary.get("topk_prob_mean", float("nan")),
    ]
    if args.router_enable_forward_ema:
        row.append(router_summary.get("ema_alpha_forward", float("nan")))
    if args.router_enable_reverse_ema:
        row.append(router_summary.get("ema_alpha_reverse", float("nan")))
    row.extend([
        router_summary.get("max_logit", float("nan")),
        (logit_cap_value if logit_cap_value is not None else float("nan")),
        router_temp_value,
        window_blocks_value,
    ])
    row.extend(expert_usage_list)
    row.extend(expert_active_list)
    metrics_csv_writer.writerow(row)

print0(code)
print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess
    try:
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    except FileNotFoundError:
        return "nvidia-smi not available."
print0(nvidia_smi())
print0("="*100)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    model_dim=args.model_dim,
    max_seq_len=max(args.train_seq_len, args.val_seq_len),
    skip_attn_layers=set(args.skip_attn_layers),
    E=args.num_experts,
    h=args.ffn_hidden,
    lb_coeff=args.lb_coeff,
    ent_coeff=args.router_entropy_coeff,
    k=args.topk,
    num_value_embeds=args.num_value_embeds,
    tie_lm_head=args.tie_lm_head,
    untie_lm_head_after=untie_lm_head_after,
    ema_alpha_fwd=args.ema_alpha_fwd,
    ema_alpha_rev=args.ema_alpha_rev,
    router_temp_init=args.router_temp_init,
    router_temp_final=args.router_temp_final,
    router_temp_power=args.router_temp_power,
    router_temp_anchor_delta_steps=args.router_temp_anchor_delta_steps,
    router_temp_anchor_ratio=args.router_temp_anchor_ratio,
    router_logit_cap_initial=args.router_logit_cap_initial,
    router_logit_cap_final=args.router_logit_cap_final,
    router_logit_cap_delta_steps=args.router_logit_cap_delta_steps,
    router_layer_peak_frac=args.router_layer_peak_frac,
    router_temp_boost=args.router_temp_boost,
    router_lb_boost=args.router_lb_boost,
    router_boost_shape=args.router_boost_shape,
    use_router_adapters=args.use_router_adapters,
    expert_activation_schedule=args.expert_activation_schedule,
    router_freeze_frac=args.router_freeze_frac,
    router_freeze_adapters=args.router_freeze_adapters,
    ema_block_size_fwd=args.ema_block_size_fwd,
    ema_block_size_rev=args.ema_block_size_rev,
    ema_window_size_fwd=args.ema_window_size_fwd,
    ema_window_size_rev=args.ema_window_size_rev,
    ema_layer_stride=args.router_ema_layer_stride,
    shared_ffn_freeze_frac=args.shared_ffn_freeze_frac,
    router_use_gumbel=args.router_use_gumbel,
    router_gumbel_schedule=args.router_gumbel_schedule,
    router_block_pos_bins=args.router_block_pos_bins,
    first_doc_tokens_N=args.first_doc_tokens_N,
    router_enable_forward_ema=args.router_enable_forward_ema,
    router_enable_reverse_ema=args.router_enable_reverse_ema,
    extra_console_logging=args.enable_extra_logging,
    extra_wandb_logging=args.enable_extra_wandb_logging,
    print_fn=print0,
).cuda()

for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

log_param_counts(model)

# collect the parameters to optimize
# ### FFNBANK MOD: include bank expert matrices in Muon; routers in AdamW with scalars+embeds.
def is_2d(p: nn.Parameter) -> bool:
    return p.ndim >= 2

attn_2d_params = []
for b in model.blocks:
    if isinstance(b.attn, CausalSelfAttention):
        attn_2d_params.append(b.attn.qkvo_w)
ffn_matrix_params = [*model.bank.W1, *model.bank.W2]
hidden_matrix_params = attn_2d_params + ffn_matrix_params

embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
head_params: list[nn.Parameter] = [model.lm_head] if model.lm_head is not None else []
adapter_params = []
if model.bank.use_adapters:
    adapter_params.extend([model.bank.adapter_scale, model.bank.adapter_bias])
scalar_params = [model.scalars]
router_params = list(model.bank.router_w) + list(model.bank.router_b)

# sanity / completeness checks
params_collections = [hidden_matrix_params, embed_params, head_params, adapter_params, scalar_params, router_params]
optimized_parameters_set = {p for params in params_collections for p in params}
assert optimized_parameters_set == {*model.parameters()}
assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)

# init the optimizer(s)
adam_param_groups = [
    dict(params=embed_params, lr=args.lr_embed, component="embed"),
    dict(params=scalar_params, lr=args.lr_scalar, component="scalar"),
    dict(params=router_params, lr=args.lr_router, component="router"),
]
if adapter_params:
    adam_param_groups.append(dict(params=adapter_params, lr=args.lr_adapter, component="adapter"))
if head_params:
    adam_param_groups.append(dict(params=head_params, lr=args.lr_head, component="head"))
optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)
muon_param_groups = []
if attn_2d_params:
    muon_param_groups.append(dict(params=attn_2d_params, lr=args.lr_muon, component="attention"))
if ffn_matrix_params:
    muon_param_groups.append(dict(params=ffn_matrix_params, lr=args.lr_muon, component="shared_ffn"))
optimizer2 = Muon(muon_param_groups, lr=args.lr_muon, momentum=0.95, rank=rank, world_size=world_size)
optimizers: list[torch.optim.Optimizer] = [optimizer1, optimizer2]
def opt_params(opt: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [p for group in opt.param_groups for p in group["params"]]
opt2params = {opt: opt_params(opt) for opt in optimizers}
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

start_step = 0
resume_path = args.resume_checkpoint
if resume_path:
    print0(f"Loading checkpoint from {resume_path}", console=True)
    checkpoint = torch.load(resume_path, map_location="cuda")
    model_state = checkpoint.get("model", {})
    if all(k.startswith("_orig_mod.") for k in model_state.keys()):
        model_state = {k.removeprefix("_orig_mod."): v for k, v in model_state.items()}
    args.approx_step_time_ms = float(checkpoint.get("approx_step_time_ms", 0))
    meta = checkpoint.get("meta", {}) or {}
    meta_checks = {
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_experts": args.num_experts,
        "ffn_hidden": args.ffn_hidden,
        "vocab_size": args.vocab_size,
    }
    for key, current_val in meta_checks.items():
        saved_val = meta.get(key, current_val)
        assert saved_val == current_val, f"Checkpoint {key}={saved_val} does not match current args ({current_val})"
    model.load_state_dict(model_state)
    ckpt_opts = checkpoint.get("optimizers", [])
    assert len(ckpt_opts) == len(optimizers), "Optimizer count mismatch in checkpoint."
    for opt, state in zip(optimizers, ckpt_opts):
        opt.load_state_dict(state)
        for group in opt.param_groups:
            group.setdefault("initial_lr", group.get("lr", 0.0))
        # ensure Muon state dtypes survive checkpoint reload
        if isinstance(opt, Muon):
            for p, st in opt.state.items():
                if not st:
                    continue
                if "mantissa" in st and st["mantissa"].dtype != torch.uint16:
                    st["mantissa"] = st["mantissa"].to(dtype=torch.uint16)
                if "momentum_buffer" in st and st["momentum_buffer"].dtype != torch.float32:
                    st["momentum_buffer"] = st["momentum_buffer"].to(dtype=torch.float32)
    start_step = int(checkpoint.get("step", -1)) + 1
    assert start_step >= 0, "Invalid checkpoint step."
    assert start_step <= args.num_iterations, "Checkpoint step exceeds configured num_iterations."
    print0(f"Resumed from checkpoint at step {start_step - 1}. Continuing from step {start_step}.", console=True)
    dist.barrier()

for param in model.parameters():
    dist.broadcast(param.detach(), 0)

print0("Compiling model...", console=True)
model: nn.Module = torch.compile(model, dynamic=False)
print0("Compile complete.", console=True)

########################################
#            Warmup kernels            #
########################################

train_micro_len = compute_train_micro_len(args.train_seq_len, args.grad_accum_steps, args.train_micro_seq_len)
effective_train_tokens = train_micro_len * args.grad_accum_steps
if effective_train_tokens != args.train_seq_len:
    print0(
        f"Adjusted train_micro_seq_len to {train_micro_len} (block-aligned). "
        f"Effective tokens per step: {effective_train_tokens} (requested {args.train_seq_len}).",
        console=True)

if args.do_model_warmup:
    print0("Warming up kernels...", console=True)
    warmup_steps = 10
    initial_state = copy.deepcopy(dict(model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers]))
    for warm_step in range(warmup_steps):
        model.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum_steps):
            inputs = targets = torch.randint(0, args.vocab_size, size=(train_micro_len,), device="cuda")
            outputs = model(inputs.to(torch.int32), targets, trainer.get_window_size_blocks(args, 0), 0, args.num_iterations)
            if isinstance(outputs, tuple):
                loss_main, loss_aux = outputs
                loss_val = float((loss_main + loss_aux).detach().item())
                main_loss = float(loss_main.detach().item())
                aux_loss = float(loss_aux.detach().item())
                loss_total = (loss_main + loss_aux) / args.grad_accum_steps
                loss_total.backward()
            else:
                loss = outputs
                loss_val = float(loss.detach().item())
                components = model.latest_loss_components
                main_loss = float(components[0].item()) if components else float("nan")
                aux_loss = float(components[1].item()) if components else float("nan")
                (loss / args.grad_accum_steps).backward()
            router_summary = summarize_router_metrics(model.latest_router_metrics or [])
            if args.enable_extra_logging:
                print0(
                    f"[warmup {warm_step + 1}/{warmup_steps} micro {micro + 1}/{args.grad_accum_steps}] "
                    f"loss={loss_val:.6f} main={main_loss:.6f} aux={aux_loss:.6f} "
                    f"{router_summary_str(router_summary, args.router_enable_forward_ema, args.router_enable_reverse_ema)}",
                    console=True)
        opt2futures = {
            opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                  for p in params if (p.grad is not None)]
            for opt, params in opt2params.items()
        }
        for opt in optimizers:
            torch.futures.collect_all(opt2futures[opt]).wait()
            opt.step()
        model.zero_grad(set_to_none=True)

    with torch.no_grad():
        model.bank.compile_warm_all_experts(d=args.model_dim, T_warm=128)

    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state
    print0("Kernel warmup complete.", console=True)

########################################
#        Training and validation       #
########################################

torch.cuda.reset_peak_memory_stats()
trainer.run_training(
    args=args,
    model=model,
    optimizers=optimizers,
    opt2params=opt2params,
    train_micro_len=train_micro_len,
    untie_lm_head_after=untie_lm_head_after,
    run_id_full=run_id_full,
    master_process=master_process,
    print0=print0,
    code=code,
    wandb_run=wandb_run,
    metrics_csv_writer=metrics_csv_writer,
    expert_usage_headers=expert_usage_headers,
    expert_active_headers=expert_active_headers,
    world_size=world_size,
    rank=rank,
    log_param_counts_fn=log_param_counts,
    start_step=start_step,
    checkpoint_save_step=args.checkpoint_save_step,
)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
if wandb_run is not None:
    wandb_run.finish()
if metrics_csv_file is not None:
    metrics_csv_file.close()
