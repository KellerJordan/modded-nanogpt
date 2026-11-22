# Track 1: #404 + exact kernel, optimizer and FP8 quantization speedups, 29.902 s (n=12; -0.145 s vs #404 on the same node)

**TL;DR:** Twelve unseeded `8×H100` runs of our `+175`-line arithmetic-identical kernel/optimizer/FP8 speedup commit (`e233546`, stacked on `#404`) break 30 s at **29.9025 ± 0.0293 s** (**−144.5 ms** vs interleaved `#404` on the same node), with all 12 runs below `3.28` (`3.27778 ± 0.00107`, `p = 8.98e-6`) and 100% byte-exact embedded source verification.

This record builds cleanly on open PR #404 (head `021718e`) with a single commit (`e233546`, 8 files, `+231 / -56` = `+175` net lines) of **pure bit-exact** kernel, optimizer, and FP8 quantization speedups.

Over all 12 unseeded runs on the same 8×H100 node, `this PR` averages **29.9025 ± 0.0293 s** vs **30.0470 ± 0.0074 s** for the 4 interleaved #404 runs (**-0.145 s** / **-144.5 ms**; paired round delta **-0.144 ± 0.019 s**, `t = -15.03`). Mean val_loss is **3.27778 ± 0.00107** (one-sided `t = 7.18`, `p = 8.98e-06` for mean < 3.28).

## Changes

Every change is strictly arithmetic-identical to #404 (verified `max_abs = 0` on GPU-0 / 1×H100 for all 5 custom Triton kernels, and verified via `AnvilBankGraphs._self_check` `bits_equal` for the fused `torch.compile` ANVIL update) and controlled via `track_1_short/exact_levers.py` (`EXACT_LEVERS=all` by default):

- **Fused MoE ANVIL cascade + zero-copy Gram** (`optim/anvil.py`, `perf/kernels/anvil_opt.py`, `moeanvil`): `moe_bank` (`margin=1.1`) uses `anvil_opt` (`pre_fused` + `polar(..., margin=bank.margin)`) instead of the `torch.compile` `anvil_cascade` fallback. In `gram_tall`, `_xtx_copy_kernel` fuses `polar_express.XTX_kernel` with the defensive `baddbmm` input copy on idle CTAs (`pid >= num_gram`), preserving bitwise identity while removing the separate memcpy.
- **Fused lane equalizer + cautious decay update** (`optim/anvil.py`, `b3opt`): `AnvilAndAdam._equalize_and_decay_update` (`@torch.compile(dynamic=False, fullgraph=True)`) fuses lane energy equalization with the BF16+mantissa cautious weight decay update without writing `v_chunk` back to HBM.
- **128×128 dual FP8 weight quantization + fused scale reduction** (`perf/kernels/mlp.py`, `perf/kernels/fp8_attention_quant.py`, `b3opt`): `quantize_mlp_weights_dual` widens tiles from `64×64` (`num_warps=4`) to `128×128` (`num_warps=8`) and replaces the ATen scale reduction with a single-launch `_reduce_weight_scales_kernel`. `quantize_dual_layout_packed_batched` uses single-load pointer selection (`SINGLE_LOAD=True`), widens tiles to `128×128`, and fuses the partial-amax scale reduction (`FUSE_SCALE=True`).
- **MoE grouped weight-gradient 128×128 tiling** (`perf/kernels/moe_mlp.py`, `moeg3bwd`): `DW_BLOCK_M, DW_BLOCK_N = (128, 128)` instead of `(128, 256)` in `grouped_dw_kernel` improves wave occupancy across 132 H100 SMs.
- **CPLM direct BF16 backward stores** (`perf/kernels/cplm.py`, `cplmbf16`): `_cplm_copy_bwd_kernel` casts `dq_acc` / `dk_acc` directly to `tl.bfloat16` on store, eliminating intermediate FP32 temporary writes and conversion copies.
- **Tail-averaging ATen kernel pre-warming** (`tail_average.py`, `tailopt`): `TailAverages.__init__` pre-warms `ship()` ATen kernels (`norm`, `lerp`, `copy_`, `mul_`, `where`) before the clock starts so CUDA lazy module loading does not land inside the timed region.

## Result

Hardware: one RunPod 8× NVIDIA H100 80GB HBM3 SXM5 node (`AP-IN-1`, 56/56 NV18 links), driver 595.112.07, CUDA 12.8, torch 2.10.0+cu128, Triton 3.6.0, Python 3.12.3. Run order: 4 interleaved rounds of [1 × `#404`, 3 × `this PR`] (16:07–16:58 UTC, 2026-11-22). Seeds were random (`TRAIN_SEED` unset), as in a normal record run; zero runs were dropped.

| arm | n | train_time (s), mean ± sd | val_loss, mean ± sd | one-sided t-test, mean < 3.28 |
|---|---|---|---|---|
| **this PR** | 12 | **29.9025 ± 0.0293** | **3.27778 ± 0.00107** | t = 7.18, **p = 8.98e-06** |
| #404, interleaved (r1–r4) | 4 | 30.0470 ± 0.0074 | 3.27948 ± 0.00246 | t = 0.43, p = 0.3490 |
| #404, all runs on this node | 10 | 30.0481 ± 0.0346 | 3.27906 ± 0.00172 | t = 1.72, p = 0.0594 |

- **Time vs #404 (interleaved, same node):** -0.145 s (-144.5 ms), Welch `t = -15.63` (`df = 13.8`, `p = 3.77e-10`).
- **Round-by-round paired delta (mean of 3 `this PR` runs minus round `#404` run, n=4 rounds):** -0.154, -0.155, -0.116, -0.154 → **-0.1445 ± 0.0192 s** (`-144.5 ms`, paired `t = -15.03`, `p = 6.39e-04`).
- **vs #404 (all 10 same-node runs):** -0.146 s (-145.6 ms), Welch `t = -10.53` (`df = 17.8`, `p = 4.48e-09`).
- **Loss:** p-values are from the one-sided Student t-test against 3.28 (`t = (3.28 − mean) / (sd/√n)`, `p = P(T_{n−1} > t)`), equivalent to `scipy.stats.ttest_1samp(..., alternative='less')`.

### NVML GPU power, energy & clock telemetry (active training window)

| arm | n | mean power (8 GPUs) | GPU energy | mean SM clock (min) | max temp |
|---|---|---|---|---|---|
| this PR | 12 | 5422.3 W | 162.14 kJ (45.04 Wh) | 1873.4 MHz (1620) | 72 °C |
| #404, interleaved (r1–r4) | 4 | 5415.6 W | 162.72 kJ (45.20 Wh) | 1877.3 MHz (1620) | 72 °C |

### Every run

Each log embeds the full source of the run. Every `this_pr/` log's embedded source matches `e233546` file for file (`60/60` files, `0` mismatches), and every `baseline_404/` log matches `021718e` (`60/60` files, `0` mismatches).

| run | arm | recorded (UTC) | train_time (s) | val_loss | energy (Wh) | log |
|---|---|---|---|---|---|---|
| base404_r1 | #404 | 16:07 | 30.040 | 3.2789 | 45.18 | `baseline_404/c7529102-4476-4199-9d11-f0e4d5c6e57f.txt` |
| prv2_r01 | this PR | 16:10 | 29.882 | 3.2786 | 44.97 | `this_pr/119a02d0-3e4c-4a38-8592-fbb6689c127e.txt` |
| prv2_r02 | this PR | 16:14 | 29.888 | 3.2780 | 44.94 | `this_pr/4f7dc4cd-5d56-4568-b617-f2cb95e4f9c4.txt` |
| prv2_r03 | this PR | 16:17 | 29.888 | 3.2756 | 45.02 | `this_pr/eb073a90-8294-4463-8534-66daa10b25b3.txt` |
| base404_r2 | #404 | 16:20 | 30.042 | 3.2831 | 45.17 | `baseline_404/12d48f73-ecec-495b-8c48-930cb8a01fba.txt` |
| prv2_r04 | this PR | 16:23 | 29.872 | 3.2767 | 45.01 | `this_pr/588ee387-1a0e-4ac1-b651-719387333f9b.txt` |
| prv2_r05 | this PR | 16:26 | 29.892 | 3.2777 | 44.99 | `this_pr/27147cc7-08ea-4fc6-b1a2-e7ed6dc1cbcb.txt` |
| prv2_r06 | this PR | 16:29 | 29.898 | 3.2786 | 45.02 | `this_pr/e78e0b98-9788-4d1c-8a07-b8a466f726f2.txt` |
| base404_r3 | #404 | 16:33 | 30.050 | 3.2779 | 45.22 | `baseline_404/dbc47130-c68d-4bb3-b7b2-9d869d7c6458.txt` |
| prv2_r07 | this PR | 16:36 | 29.906 | 3.2794 | 45.10 | `this_pr/03b13578-3a1e-44eb-8a07-1fe83c4432a0.txt` |
| prv2_r08 | this PR | 16:39 | 29.909 | 3.2775 | 45.06 | `this_pr/4752a761-2062-4e12-b0fa-4b39f4dcd8da.txt` |
| prv2_r09 | this PR | 16:42 | 29.988 | 3.2775 | 45.17 | `this_pr/200946ca-9aab-453c-a3c2-7b8137f6fac1.txt` |
| base404_r4 | #404 | 16:45 | 30.056 | 3.2780 | 45.23 | `baseline_404/119f6ebb-0d74-4b0a-b71a-e7b12e0166d8.txt` |
| prv2_r10 | this PR | 16:48 | 29.898 | 3.2783 | 45.08 | `this_pr/6dd1b46d-6ff4-4966-856f-3b890c8b4bce.txt` |
| prv2_r11 | this PR | 16:51 | 29.914 | 3.2788 | 45.09 | `this_pr/d62f6647-278d-4e35-b190-4978e513de86.txt` |
| prv2_r12 | this PR | 16:55 | 29.895 | 3.2767 | 45.04 | `this_pr/13c94c92-0c7c-4014-a731-e1edf306210d.txt` |

## Exactness verification (`1×H100` / GPU-0)

Every new lever in this PR was gated for strict bitwise parity (`max_abs = 0`) against `#404` before `8×H100` training runs:

| Lever / Kernel | Parity vs `#404` | GPU-0 / `1×H100` Latency | Notes |
|---|---|---|---|
| `MOE_ANVIL_OPT` (`anvil_opt._xtx_copy_kernel` + `margin=1.1`) | **PASS (`max_abs = 0`)** | `516.0 → 438.0 µs` | Fuses `moe_bank` onto `anvil_opt` without `baddbmm` defensive copies |
| `B3_OPT` (`fp8_attention_quant` `128×128`, `SINGLE_LOAD`, `FUSE_SCALE`) | **PASS (`max_abs = 0`)** | `196.2 → 122.3 µs` | Single-load partial amax + fused scale reduction (`15 → 6` launches) |
| `CPLM_BF16` (`cplm._cplm_copy_bwd_kernel` direct `bf16`) | **PASS (`max_abs = 0`)** | `174.7 → 122.9 µs` (`T=49k`) | Direct `bf16` store eliminates intermediate FP32 cast/buffer |
| `B3_OPT` (`mlp.quantize_mlp_weights_dual` `128×128` + Triton scale reduce) | **PASS (`max_abs = 0`)** | `130.5 → 95.8 µs` | Eliminates ATen `amax` launch overhead |
| `MOE_G3_BWD_PTX` (`moe_mlp.grouped_dw_kernel` `128×128`) | **PASS (`max_abs = 0`)** | `723.1 → 702.5 µs` | Improves wave occupancy across 132 H100 SMs |
| `B3_OPT` (`anvil._equalize_and_decay_update`) | **PASS (`AnvilBankGraphs._self_check`)** | Fused across all 4 banks | Fuses lane equalizer + cautious decay preserving intermediate `bf16` rounding without writing `v_chunk` to HBM |
| `TAIL_OPT` (`tail_average.TailAverages.__init__` pre-warm) | **PASS (0 diff)** | Init-only pre-warm | Pre-warms `ship()` ATen kernels before clock start |

## Reproduce

```bash
git checkout e233546      # = #404 head 021718e + this PR
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All exact levers are enabled by default (`EXACT_LEVERS=all`). To restore `#404` exactly on the same checkout:

```bash
EXACT_LEVERS=fa3copy,rsearly,bankinplace,prescatter torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Disclosures

- **Hardware.** Runs used one RunPod `8×H100 80GB HBM3` SXM5 node (`AP-IN-1`), not PrimeIntellect/GKE. Absolute times can vary across providers and driver stacks; the same-node interleaved deltas are the controlled comparison.
- **Exactness & nondeterminism.** All 5 custom Triton kernel changes passed strict `max_abs = 0` bitwise parity vs `#404` on GPU-0 / 1×H100, and the fused `torch.compile` ANVIL update preserves the intermediate `bf16` rounding and passes `AnvilBankGraphs._self_check` (`bits_equal`). End-to-end training inherits `#404`'s FA3 backward / atomic nondeterminism and uses random seeds.
- **Completeness.** All `12` runs of `this PR` and all `4` interleaved `#404` runs from the certification session are reported; none was dropped.

## Taken from other PRs

- **#404** by `@caraxesml`: exact startup/tail systems speedups (`BackgroundCanonicalMask` `posix_spawn`, `ValPrefetch`, `FirstShardStart`, `tied_grad_fold`) and truncated attention backward (`FlashAttnVarlenTruncBwd`) on short-window layers in stages 1–2, on top of which this PR stacks cleanly.
- **#396 / #397** by `@atveit`: exact systems levers framework (`track_1_short/exact_levers.py`: `fa3copy`, `rsearly`, `bankinplace`, `prescatter`).
- **#393** by `@francesco-prignano`: base Track 1 architecture, MoE ANVIL banks, CPLM, normalized n-gram table, and CUDA graph harness.
