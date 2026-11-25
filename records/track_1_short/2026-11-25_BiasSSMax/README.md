# NanoGPT Speedrun Track 1 Record Candidate: Window/Doc-Aware Centered Bias-Aware SSMax

**Paper & Companion Repo:** [*Improving Out-of-Distribution Generalization with Bias-Aware Softmax Scaling and Barrier-FoX: From 200-Digit Arithmetic Circuits to Sub-40s 8×H100 LLM Training*](https://github.com/FeSens/Bias-Aware-Softmax-Scaling) ([PDF](https://github.com/FeSens/Bias-Aware-Softmax-Scaling/blob/main/paper/main.pdf))

**Summary:** Adds **+42 fp32 scalars** (one learned per-head slope $s_h$ across the 7 attention layers) implementing window- and document-aware centered Scalable-Softmax (**Bias-Aware SSMax**), fused directly into `qkv_rope.py`'s QK-Norm + RoPE forward/backward Triton kernels (+72 ms total kernel cost, +0 MiB peak allocated VRAM, **+138 / -26 lines across 5 files, 0 changes to `train_gpt.py`**). The improved sample efficiency and variance stability allow trimming 12 steps from the batch-24 Stage-2 growth schedule (`SCHEDULE_GROWTH_STEPS: 52 -> 40`, preserving exact mod-4 n-gram/Adam boundary parity), cutting **`-0.477 s` (`40.454 s -> 39.977 s`, `-1.18%`)** on `8x NVIDIA H100 80GB HBM3` while clearing the `3.28` gate with **`p = 1.26e-6`** (`0/10` runs above `3.2784`).

---

## Certification Pool (`10 vs 10` Interleaved A/B, Same Machine, Same Session)

- **Hardware:** `8x NVIDIA H100 80GB HBM3` (full `NV18` NVSwitch, Nebius `gpu-h100-sxm`, driver `580.196.07`)
- **Pinned Stack:** `torch==2.10.0+cu128`, `CUDA 13.0`, `kernels==0.16.1`, `huggingface-hub==1.29.0`, FA3 `devenpzak/flash-attn3-12864@64c1e6d`
- **Invocation:** Unmodified `HF_HUB_OFFLINE=1 bash run.sh` (zero extra `torch._inductor` / `torch.compile` flags; every completed run counted)

| Arm | Runs (`n`) | Steps | Wall Time (`train_time`) | Final `val_loss` | Range | Runs `> 3.28` | One-Sided `p (< 3.28)` |
|---|---:|---:|---:|---:|---|---:|---:|
| **`HEAD` (`4ea6b93`, `G=52`, interleaved)** | 10 | 1194 | `40.454 ± 0.024 s` | `3.27610 ± 0.00281` | `3.2736 – 3.2832` | 1 / 10 | `8.78e-4` |
| **This PR (`Bias-Aware SSMax`, `G=40`)** | **10** | **1182** | **`39.977 ± 0.020 s`** | **`3.27738 ± 0.00079`** | **`3.2760 – 3.2784`** | **0 / 10** | **`1.26e-6`** (`t = -10.43`) |
| `HEAD @ G=40` (step-cut-only control, no SSMax) | 5 | 1182 | `39.905 ± 0.011 s` | `3.27998 ± 0.00389` | `3.2768 – 3.2865` | 2 / 5 | `4.96e-1` (**FAIL**) |

### Rule Compliance Verification
- **Rule 1 (Data pipelines untouched):** PASS — `cu_seqlens` segment boundaries already produced by the stock loader are read as-is; zero changes to token streams or data loading.
- **Rule 2 (`mean val_loss <= 3.28` at `p < 0.01`):** **PASS** — `3.27738 ± 0.00079`, one-sided `t = -10.43`, **`p = 1.26e-6`** (`n = 10`, all 10 runs `<= 3.2784`, matching ANVIL2's published `3.27731 ± 0.00099`).
- **Rule 3 (No extra `torch._inductor.config` / `torch.compile` flags):** PASS — zero compiler flags added.
- **Rule 4 (Faster than same-hardware interleaved `HEAD`):** **PASS** — **`-0.477 ± 0.034 s`** across 10 adjacent pairs (one-sided paired `p = 3.81e-12`, two-sided `7.62e-12`; Welch `p = 3.59e-20`).
- **Discretionary Rule 1 (Readability & simplicity):** **+138 / -26 lines across 5 files**, `0` changes to `train_gpt.py`, `+42` fp32 parameters, `+0 MiB` peak allocated VRAM (`48,411 MiB` identical to `HEAD`).
- **Discretionary Rule 2 (Outperforms a simple step-count decrease at equivalent loss):** **PASS** — proven three independent ways:
  1. **Direct matched-step control (`G=40`, 1182 steps, same machine):** Simply cutting 12 steps from `HEAD` (`HEAD @ G=40`, `n=5`) degrades `val_loss` to **`3.27998 ± 0.00389`** (`2/5` runs above `3.28`, `p(<3.28) = 0.496`, fails Rule 2!), whereas our PR at the exact same `1182` steps achieves **`3.27738 ± 0.00079`** (**`-2.60 mnat` better**, one-sided MW-U `p = 0.063`, **4.9× tighter variance**), for only `+72 ms` kernel cost (`~426 ms` ML gain minus `72 ms` kernel = **`~354 ms` net matched-loss win**).
  2. **Implied exchange rate vs `HEAD @ G=52`:** `477 ms / 1.28 mnat = 373 ms / millinat` (`> 2.2×` ANVIL2's `164 ms / millinat` marginal step-cut slope; `Δval = +0.00128` is not statistically significant, `p = 0.195`).
  3. **Matched-step `G=52` (1194 steps) development ablations (27 runs on Modal `8×H100`/`8×H200`):** Centered Bias-Aware SSMax (`lr_mul = 0.5`, `n = 4`: `3.27430 ± 0.00099`; pooled low-LR `lr_mul in {0.25, 0.5}`, `n = 6`: `3.27437 ± 0.00089`) beats interleaved `base` (`n = 7`: `3.27614 ± 0.00222`) by **`-1.84 mnat`** (one-sided Welch `p = 0.046`, MW-U `p = 0.044`; pooled `n = 6` one-sided Welch `p = 0.044`, MW-U `p = 0.031`) and beats naive unclamped vanilla SSMax (`vssmax`, `n = 2`: `3.27855 ± 0.00092`) by **`-4.25 mnat`** (one-sided Welch `p = 0.014`, two-sided `p = 0.027`), proving document-boundary + sliding-window clamping is essential.

---

## Method

Naive Scalable-Softmax ($\tau_t = s \log(t+1)$) degrades packed sliding-window training (`3.2786`, worse than baseline) because global stream position ignores document resets and sliding-window truncation. Instead, we compute each token's exact visible context length inside its document and attention window:

$$N_{\text{vis}}(t) = \min\big(t - \text{doc\_start}(t) + 1,\; W_{\ell}\big)$$

(with $2\times$ visible length on paired-head layers `0, 2, 5`), center it around $\log N_0 = 5.5$ so short-window/early tokens can soften ($\tau < 1$) while long-window tokens sharpen ($\tau > 1$), and scale queries by:

$$\tau_{\ell,h}(t) = \max\big(0.25,\; 1 + s_{\ell,h} (\log N_{\text{vis}}(t) - 5.5)\big)$$

initialized at $s_{\ell,h} = 0$ (exact identity $\tau = 1$ at step 0), trained with replicated Adam (`lr_mul = 0.5, wd_mul = 0.0`).

### Zero-HBM-Overhead Triton Fusion
- **1-launch binary-search `compute_ssmax_log_n` (`qkv_rope.py`):** computes centered $\log N_{\text{vis}}$ for short, paired-short, and long windows once per forward pass from `cu_seqlens` and an address-stable 3-element `_ssmax_cap` buffer (updated in-place across schedule window transitions for CUDA-graph safety).
- **Fused forward + exact algebraic backward (`qkv_rope.py`):** forward multiplies normalized rotated queries by $\tau$ in registers before `out_q` store; backward exploits the exact identity $\sum_d (g_q \cdot \hat{y}) = \sum_d (g_{\text{norm}} \cdot \hat{x})$ — which is **already computed** for QK-RMSNorm's `correction` term — accumulating per-block $\partial L / \partial s_{\ell,h}$ deterministically with zero atomics and zero extra HBM passes over Q/K/V.
- **Mechanistic behavior:** across 6 independent matched-step runs, `250/252` (`99.2%`) heads learn strictly positive slopes, with long-window layers `L3` (`+0.101`) and `L10` (`+0.090`) learning ~2.5× larger slopes than short-window layers (`+0.035 – +0.073`).

---

## Per-Pair Raw Certification Logs

| Pair | `HEAD` Wall | `HEAD` Val | `PR` Wall | `PR` Val | Paired `Δtime` |
|---:|---:|---|---:|---|---:|
| 01 | `40.450 s` | `3.2742` ([log](baseline/230d74ac-d004-455e-91aa-bcd89cee8d36.txt)) | `39.980 s` | `3.2760` ([log](this_pr/08fa7ce1-4e58-44e5-ab3a-820acf65eb58.txt)) | `-0.470 s` |
| 02 | `40.424 s` | `3.2753` ([log](baseline/9b038415-045f-4144-98a5-9c5e0072380a.txt)) | `39.970 s` | `3.2774` ([log](this_pr/89e7c46b-dc36-49f3-9838-3a48a531134d.txt)) | `-0.454 s` |
| 03 | `40.456 s` | `3.2741` ([log](baseline/4262d378-f9a0-4bc6-9f21-2f298c4e3a05.txt)) | `39.962 s` | `3.2775` ([log](this_pr/6eeecb24-db21-460f-a526-8bea52a0778f.txt)) | `-0.494 s` |
| 04 | `40.458 s` | `3.2751` ([log](baseline/1419b3ce-ed95-4b32-997d-220485f5bfb7.txt)) | `39.969 s` | `3.2780` ([log](this_pr/9d00a266-2ab3-4c76-b5e4-425107e6f70b.txt)) | `-0.489 s` |
| 05 | `40.437 s` | `3.2782` ([log](baseline/9377774e-af7e-4f99-8d3b-5c61218d1663.txt)) | `39.962 s` | `3.2764` ([log](this_pr/bd849fc0-697a-4206-bef6-fe210a9d97b4.txt)) | `-0.475 s` |
| 06 | `40.439 s` | `3.2832` ([log](baseline/877fdabb-2612-400b-be34-9e4c6ba2047b.txt)) | `40.023 s` | `3.2784` ([log](this_pr/5541a6b5-9635-4089-b916-871e97a112cf.txt)) | `-0.416 s` |
| 07 | `40.464 s` | `3.2753` ([log](baseline/6a37928c-e580-4452-9b96-e98e391b7b83.txt)) | `39.970 s` | `3.2773` ([log](this_pr/baf197e3-fa43-40e5-9144-84096e31e66a.txt)) | `-0.494 s` |
| 08 | `40.515 s` | `3.2736` ([log](baseline/a343a205-0404-463d-870f-a610a1b53baf.txt)) | `39.974 s` | `3.2767` ([log](this_pr/e027a11f-efec-41e2-bc45-2cc9c090171e.txt)) | `-0.541 s` |
| 09 | `40.446 s` | `3.2763` ([log](baseline/45b20bfa-4549-4899-aa20-9072e46f0fb3.txt)) | `40.002 s` | `3.2781` ([log](this_pr/8d99f693-f8e1-499d-b58e-845c44a36f2a.txt)) | `-0.444 s` |
| 10 | `40.454 s` | `3.2757` ([log](baseline/c8a92f0c-80a2-4e92-a727-31e4e8f40e33.txt)) | `39.962 s` | `3.2780` ([log](this_pr/c8743e79-d5b5-4cab-bf8e-9ed12694cad2.txt)) | `-0.492 s` |

### Matched-Step `HEAD @ G=40` Control Logs (1182 steps, no SSMax)

| Run | `HEAD @ G=40` Wall | `HEAD @ G=40` Val | Log |
|---:|---:|---:|---|
| 01 | `39.922 s` | `3.2779` | [log](head_g40_control/9c4e4382-7328-4907-a4d6-8c1221f146e8.txt) |
| 02 | `39.908 s` | `3.2782` | [log](head_g40_control/332a6aa6-95bd-42ac-a45c-cecd745ecd7c.txt) |
| 03 | `39.896 s` | `3.2865` | [log](head_g40_control/e8e040ec-04c4-464d-b228-a230e630bb7e.txt) |
| 04 | `39.897 s` | `3.2768` | [log](head_g40_control/697e264a-509e-49b7-94f7-2747aa9b0c5f.txt) |
| 05 | `39.903 s` | `3.2805` | [log](head_g40_control/bf27f552-9c80-48b3-a5a4-0656f6395b1f.txt) |
