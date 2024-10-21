"""
Runs a coord check on the model defined in config.py.
Data is dummy.
"""

import os
from contextlib import nullcontext

import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch._inductor.config as torch_ind_config

from coord_check import get_coord_data, plot_coord_data
from train_ngpt2 import GPT, GPTConfig

# --------------------------

output_dir = ""

use_mup = True
widths = [64, 128, 256, 512, 768] # check that for all these widths, d_model is divisible by d_head
mup_base_width = 64
n_layers = 4
d_head = 64
lr = 2e-3
batch_size = 16
ctx_len = 256
max_value = 100

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

class RandomDataset(Dataset):
    def __len__(self):
        return 9999999

    def __getitem__(self, idx):
        data = torch.randint(low=0, high=max_value, size=(batch_size, ctx_len))
        x = data[:, :-1].int()
        y = data[:, 1:].long()
        return x, y

def lazy_model(width):
    config = GPTConfig(vocab_size=max_value, n_layer=n_layers, n_head=width//d_head, n_embd=width)
    if use_mup:
        config.mup_width_mult = width / mup_base_width
    return GPT(config).to(device)

models = {width: (lambda: lazy_model(width)) for width in widths}

dataset = RandomDataset()
loader = DataLoader(dataset, batch_size=None, shuffle=True)
iter_ = iter(loader)

if not use_mup:
    optcls = lambda model: model.configure_optimizer(lr, weight_decay=0., betas=(0.9, 0.95))
else:
    optcls = lambda model: model.configure_optimizer_mup(lr, weight_decay=0., betas=(0.9, 0.95))

df = get_coord_data(models, iter_, optcls, dtype_ctx, nsteps=10)

if use_mup:
    name = "ngpt_mup.png"
else:
    name = "ngpt_no_mup.png"

plot_coord_data(df, legend="auto", save_to=os.path.join(output_dir, name))
