# MUDDFormer connections (GPT-2 Medium / `train_gpt_medium.py`)

Add MUDD (Multiway Dynamic Dense Connections, [Xiao et al. 2025](https://arxiv.org/abs/2502.12170)) to the **last two transformer blocks** of the Medium speedrun model (`GPT` with **16 layers**, **d_model=1024**, **8 heads × 128** head dim). The idea matches the Small-track PR: trim routing cost, reuse one small MLP for several cheap “extra outputs,” and fold legacy scalars / backout into learnable MUDD biases so we can squeeze **loss per step** without paying much wall-clock.

## What the code does today (N = 16)

MUDD still means: an MLP on the current hidden produces per-token weights over a small set of earlier representations; those weights route adds into V and the residual (we skip separate Q/K MUDD paths from the paper). With **N = 16**, the active indices are **`i = 14` (N−2)** and **`i = 15` (N−1)** — not layers 9–10 as on the 11-layer Small model.

### Layer **14** (`num_layers - 2`)

- **Sources for `v_mudd` and the layer-14 residual delta:** `{x0, h_backout, x}` — three slots. Here **`h_backout`** is the hidden snapshot taken at **`backout_layer = 11`** (post-MLP), same role as the old “h7 / early deep” snapshot on Small, but at the depth Medium’s block mask / skip layout uses (`train_gpt_medium.py` around `h_backout_snap` / `if i == backout_layer`).
- **`v_mudd`** is shaped like V and added after the QKV matmul (`1.15 *` blend of the three sources, then view to `(B, T, heads, head_dim)`).
- **Residual on layer 14:** fused self-reference `(1 + m_self) * x + m0 * x0 + m_bo * h_backout` (same trick as Small v2’s `(1 + m_r9) * x + …`).
- **Extra columns from the same `dense_w1[0]` → `dense_w2[0]` stack** (width `L + 5` with `L = 3`):
  - **`next_ve_gate`** for **layer 15** attention (replaces that layer’s static `ve_gate_w` path for the value-embed gate — analogous to “layer 10” on Small).
  - **Six dynamic scalars for layer 15:** `resid_attn`, `post_attn`, `resid_mlp`, `post_mlp`, `x0_lambda`, `bigram_lambda` (replace the static per-layer tensors for **the last layer only**), with biases chosen so the effective init matches the old scalars (`dense_bs` init in `_init_mudd`).

### Layer **15** (`num_layers - 1`) — **post main loop**

- There is **no** `elif i == num_layers - 1` inside the compiled layer loop. The final residual MUDD runs **once after** `for i in range(self.num_layers)`, then `norm` → logits / fused loss.
- **Sources (5):** `{x0, h_backout, h14, ve_bank0, skip_connection}`  
  - **`h14`:** post-MLP snapshot at layer 14 (`h_n2_snap`).  
  - **`ve_bank0`:** `ve_list[0]` → **`ve[0]`** in the five-bank tensor, i.e. the VE bank attached to **layer 0** in this model’s `ve_list` wiring (Small’s post-loop mix used **`ve[1]`** for the layer-1 bank — different index because the layer↔bank map differs between tracks).  
  - **`skip_connection`:** snapshot at **layer 4** (after that block’s MLP) — Medium uses **layer 4** as the long-window skip tap for this path (Small used layer 3 in the analogous v3 write-up).
- **No self-reference** in that five-way sum: plain `x += m0*x0 + …` style mix; **`dense_bs[1, 1, 1] = -4.34`** absorbs the old scalar **backout** on the `h_backout` slot (same idea as Small: effective initial `≈ -0.5 * h_backout` after `_mudd_scale`).

### Optimizer / init (unchanged in spirit from Small)

- **`dense_w1` / `dense_w2` / `dense_bs`:** replicated Adam, **`lr_mul=0.25`**, `betas=(0.9, 0.99)`, no WD on `dense_bs`; **`dense_w2` starts at zero** so MUDD starts as a no-op except where biases explicitly encode static inits (gates / backout slot).
- **`_mudd_scale = 0.2 / sqrt(L)`** with **`L = 3`** on the first MUDD block.


### Results
losses = [2.9161, 2.9165, 2.9176, 2.9179, 2.9172, 2.9171]
times = [860467, 860193, 860109, 860808, 860679, 860614]
```bash
print("p=%.4f" % scipy.stats.ttest_1samp(medium_losses, 2.92, alternative="less").pvalue)
# p=0.0001

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0007), tensor(2.9171))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(277.5980), tensor(860478.3125))
```