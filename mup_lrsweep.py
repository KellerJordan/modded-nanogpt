"""
Runs a LR sweep on different widths on short training runs to see if muP works well (ie if LR transfers across widths).
Data is wikitext, vocab_size=256 to speed up training.
Adapted from the example in https://github.com/graphcore-research/unit-scaling.
They use it to benchmark u-muP, a newer version of muP.
"""

import os
from typing import *
from contextlib import nullcontext
import random
import math

import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch._inductor.config as torch_ind_config

from train_gpt2 import GPT, GPTConfig

# --------------------------

filename = "gpt2_mup_fulltest.results.json"
fig_name = "gpt2_mup_fulltest.png"

type_to_lr_range = {
    #"SP": [2**n for n in range(-11, -7 + 1)],
    "μP": [2**n for n in range(-15, -13 + 1)],
}

ctx_length = 256

# --- model parameters ---
mup_base_width = 768
widths = [64, 768] # check that for all these widths are divisible by d_head
n_layers = 4
d_head = 64

batch_size = 128

vocab_size = 256

num_iters = 4000
lr_warmup_iters = 200
lr_warmdown_iters = 800

device = "cuda"
dtype = "bfloat16"
use_torch_compile = False

# --------------------------

seed = 123456789 + 0

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"
torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
dtype_ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, torch_dtype))

if use_torch_compile:
    if hasattr(torch_ind_config, "coordinate_descent_tuning"):
        torch_ind_config.coordinate_descent_tuning = True

def wsd_schedule(warmup_iters, decay_iters, num_iters, start_iter=0):
    def schedule(iter):
        iter = start_iter + iter
        if iter > num_iters:
            return 0.
        # linear warmup
        if iter < warmup_iters:
            return iter/warmup_iters
        # hold
        elif iter < (num_iters - decay_iters):
            return 1.
        # decay (with 1-sqrt)
        else:
            if decay_iters==0:
                return 1.
            return 1 - math.sqrt((iter - (num_iters - decay_iters))/decay_iters)
    return schedule

# ------- data -------
dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
data = torch.frombuffer(bytearray("".join(dataset["text"]), encoding="utf8"), dtype=torch.uint8)
def batches():
    for _ in range(num_iters):
        yield torch.stack([
            data[i:i + ctx_length].to(device=device, dtype=torch.long)
            for i in torch.randint(0, len(data) - ctx_length, size=(batch_size,))
        ])

def run_experiment(type_: Literal["SP", "μP"], width: int, lr: float) -> List[Dict[str, Any]]:
    if type_ == "μP":
        use_mup = True
        
    elif type_ == "SP":
        use_mup = False

    config = GPTConfig(vocab_size=vocab_size, n_layer=n_layers, n_head=width//d_head, n_embd=width)
    if not use_mup:
        config.mup_width_mult = 1
    else:
        config.mup_width_mult = width / mup_base_width

    model = GPT(config).to(device)
    model = torch.compile(model)
    model.train()

    if not use_mup:
        optim = model.configure_optimizer(learning_rate=lr, weight_decay=0., betas=(0.9, 0.95))
    else:
        optim = model.configure_optimizer_mup(learning_rate=lr, weight_decay=0., betas=(0.9, 0.95))
    
    scheduler = lr_scheduler.LambdaLR(optim, wsd_schedule(warmup_iters=lr_warmup_iters, decay_iters=lr_warmdown_iters, num_iters=num_iters, start_iter=0))

    def run_step(batch):
        x = batch[:, :-1]
        y = batch[:, 1:].contiguous()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with dtype_ctx:
            _, loss = model(x, y, return_logits=False)

        loss.backward()

        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

        optim.step()
        scheduler.step()
        optim.zero_grad()
        return loss

    log = []
    log2lr = torch.tensor(lr).log2().item()
    progress = tqdm(enumerate(batches()), desc=f"{type_:>4}, width={width}, lr=2^{log2lr:<5.0f}")
    for step, batch in progress:
        loss = run_step(batch)
        log.append(dict(step=step, loss=loss.item()))
        if (step + 1) % 100 == 0:
            progress.set_postfix_str(f"loss = {loss.item():.2f}")
    return pd.DataFrame.from_dict(log).assign(type=type_, width=width, lr=lr)

# ---------------

if os.path.exists(filename):
    with open(filename, "r") as f:
        existing_results = pd.read_json(f)
else:
    existing_results = pd.DataFrame()

new_results = pd.concat([
        run_experiment(type_=type_, width=width, lr=lr)
        for type_, lrs in type_to_lr_range.items()
        for width in widths
        for lr in lrs
]).reset_index(drop=True)
combined_results = pd.concat([existing_results, new_results]).reset_index(drop=True)
combined_results.to_json(filename)

combined_results["loss"] = combined_results["loss"].fillna(2.)
df_final = combined_results.groupby(["type", "width", "lr"])["loss"].apply(lambda g: min(g.iloc[-50:].mean(), 2.)).reset_index()

g = sns.relplot(data=df_final.pipe(lambda d: d.assign(width=d.width.apply(str))),
                y="loss", x="lr", hue="width", col="type",
                kind="line", facet_kws=dict(sharex=False), height=4)
for type_, ax in g.axes_dict.items():
    ax.set_title(type_)
    ax.set_xscale("log", base=2)

g.savefig(fig_name, dpi=600)
