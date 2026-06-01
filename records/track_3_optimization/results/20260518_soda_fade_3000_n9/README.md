# Track 3 result: SODA fade at 3000 steps

This directory contains the submission bundle for
`records/track_3_optimization/train_gpt_contra_soft_soda_fade_official.py`.

Submitted checkpoint: **3000 steps**.

Runs included: seeds **0 through 8**, all completed runs in this result set.
Each file under `code_logs/` is a self-contained run log containing the full
train script plus validation output. The `stdout/` files are plain stdout
captures used to map seeds to generated code-log filenames.

Run command:

```bash
torchrun --standalone --nproc_per_node=4 records/track_3_optimization/train_gpt_contra_soft_soda_fade_official.py --seed <seed>
```

Hardware used for these logs: 4x NVIDIA H200 Vast.ai instances. We tried cheaper
H100 Vast.ai instances first, but the available H100 hosts failed during
container/GPU startup or were unhealthy over SSH/network; H200 was used as the
first healthy official-shaped machine. Track 3 is a step-count benchmark; the
dataset, batch size, architecture, and one forward/backward pass per step are
unchanged. Total Vast.ai spend for the official H200 runs was $55.95.

## Use of AI

This result was produced collaboratively by @wakamex and Codex running GPT-5.5
xhigh.

@wakamex drove the research direction and experimental judgment: selecting the
SODA / optimistic-dual-averaging idea to try, asking whether it could be used as
a wrapper on top of the current best Contra-Muon-to-Soft-Muon + SOAP result,
requesting an as-close-as-possible local RTX 3090 trial before spending on
remote GPUs, requesting official-shaped full runs rather than relying on the
local proxy, setting the cost/compute constraints, providing Vast.ai access, and
deciding to target a stronger 3000-step submission after the first official runs
showed a 3020-step result.

Codex implemented and ran the experiments: adapting the current best Track 3
script into `train_gpt_contra_soft_soda_fade_official.py`, adding the
hidden-matrix anchor correction and cosine fade schedule, adding local
single-GPU trial knobs, running the RTX 3090 feasibility trial, preparing the
Vast.ai environment, launching and monitoring the official-shaped seed runs,
computing the Track 3 significance condition, stopping the extra run once
completed seeds 0-8 were enough for the requested 3000-step claim, copying the
self-contained logs, and preparing this submission bundle.

Full chat-log is available at <a href="https://mihaicosma.com/soda_run.html">https://mihaicosma.com/soda_run.html</a>.

## Methodology

A short three-way proxy control helped check that SODA's early signal was not
just weight-decay removal. At 120 proxy steps, the same-seed losses were: Muon
with `wd=.025`, 6.14614; Muon with `wd=0` and no SODA, 6.14974; and SODA-hidden
Muon, 6.08696.

After early feasibility checks showed that the SODA anchor path survived the
memory constraints of a 24 GB card, we ran full 3040-step, official-global-batch
local trials with `microbatch_seqs=8` and `no_compile=True`. This was the closest
stable 3090 configuration we found: `microbatch_seqs=8` was the largest fitting
microbatch, and compiled `mbs=8` fit memory but produced NaNs at step 2 on the
local `torch==2.10.0+cu128` stack. The always-on SODA trial reached step 3040
with val loss 3.27906 after 52384.035s of training time, about 14.6 hours. The
later SODA-fade local trial reached step 3040 with val loss 3.27651 after
52318.135s, again about 14.5 hours. The official result below does not use those
local logs as evidence, but those complete curves motivated the official-shaped
H200 runs.

The motivation for the SODA variant was to test whether optimistic dual
averaging could be used as a conservative wrapper around the then-current best
optimizer rather than as a wholesale replacement. By the time this was tried,
we had learned that short local proxies were not reliable for ranking official
runs, so the local 3090 work was used in a narrower way: first as an
implementation and stability check, and then as a full-shape curve comparison
against the exact upstream run we forked from:
`records/track_3_optimization/results/20260509_contra_soft_muon/03c36e81-e2e5-4916-bf16-0141999b1dbb.txt`.

The always-on local SODA run was ahead of the source Contra-Muon-to-Soft-Muon +
SOAP run through most of the middle of training, roughly steps 625 through 2750,
but the source run caught and passed it between steps 2750 and 2875. At the
target-region checkpoints the always-on SODA run was behind the source run: at
step 3000 it was 3.28102 versus 3.27916, and at step 3040 it was 3.27906 versus
3.27665. This suggested that the SODA anchor was useful as a mid-run regularizer
or trajectory bias, but was acting as drag during the final approach.

This choice was also inspired by the upstream champion's own scheduled optimizer
components. The source run does not keep one Muon variant fixed for the whole
run: it linearly moves from Contra Muon to normal Muon by step 2000, then ramps
from normal Muon toward Soft Muon from step 2500 to step 3010. Our SODA fade used
the same broad lesson, applying the anchor during the region where the local
curve said it helped, then turning it off before the late target-loss region.

The schedule was therefore added to preserve the useful part of the observed
curve while removing the anchor before the benchmark endpoint. In the code, each
hidden matrix stores `soda_anchor` as a clone of its initial value. At each Muon
step, before applying the Muon update, the parameter is mixed toward that anchor
with `lambda = SODA_LAMBDA_SCALE * fade(step) / (step + 2)`.
`SODA_LAMBDA_SCALE` is 1.0, `fade(step)` is 1 before step 2000, follows a
cosine decay from step 2000 through step 2750, and is 0 after step 2750. The
local fade run then showed the intended effect: it reached step 3000 at 3.27891
and step 3040 at 3.27651, versus 3.28102 and 3.27906 for always-on SODA, making
it plausible enough to test on official-shaped H200 runs.

## Statistical Check

Track 3 requires `(3.28 - mu) * sqrt(n) >= 0.004`, where `mu` is the mean val
loss over `n` non-cherry-picked runs at the same selected step.

| Step | Mean | n | Score | Pass |
| ---: | ---: | ---: | ---: | :---: |
| 2875 | 3.28838000 | 9 | -0.02514000 | no |
| 3000 | 3.27779333 | 9 | 0.00662000 | yes |
| 3010 | 3.27711444 | 9 | 0.00865667 | yes |
| 3020 | 3.27648333 | 9 | 0.01055000 | yes |
| 3030 | 3.27591667 | 9 | 0.01225000 | yes |
| 3040 | 3.27541667 | 9 | 0.01375000 | yes |

Earliest shared eligible step: **3000**.

## Per-Run Values

| Seed | Step 3000 | Step 3020 | Step 3040 | Self-contained log | Stdout capture |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 3.27895 | 3.27763 | 3.27655 | `code_logs/a3341c9d-b2a1-4fcf-91e9-8004aedcad83.txt` | `stdout/seed0_stdout.txt` |
| 1 | 3.27796 | 3.27667 | 3.27560 | `code_logs/0fdbe617-37dd-470d-b033-b6ec24fc13fc.txt` | `stdout/seed1_stdout.txt` |
| 2 | 3.27807 | 3.27675 | 3.27569 | `code_logs/942b4030-ddf0-4e39-9e4a-bac12d001cae.txt` | `stdout/soda_fade_official_seed2_20260518_150348.out` |
| 3 | 3.27907 | 3.27773 | 3.27668 | `code_logs/e5fd4f26-49e3-4251-a0a5-b4262253b08c.txt` | `stdout/soda_fade_official_seed3_20260518_152315.out` |
| 4 | 3.27851 | 3.27720 | 3.27615 | `code_logs/d8ed0aa2-4ac6-460f-bf8b-02cbac06f23a.txt` | `stdout/soda_fade_official_seed4_20260518_154154.out` |
| 5 | 3.27575 | 3.27447 | 3.27340 | `code_logs/94252eb3-3b09-4ed8-a4bc-f9cb643017bf.txt` | `stdout/soda_fade_official_seed5_20260518_160037.out` |
| 6 | 3.27730 | 3.27602 | 3.27493 | `code_logs/6d8836b6-7d8b-4f13-a3b1-a56e73e97998.txt` | `stdout/soda_fade_official_seed6_20260518_161928.out` |
| 7 | 3.27718 | 3.27586 | 3.27478 | `code_logs/af8cf7ca-f69a-47d4-9f6c-35be957197d3.txt` | `stdout/soda_fade_official_seed7_20260518_163819.out` |
| 8 | 3.27735 | 3.27602 | 3.27497 | `code_logs/ae3f35c8-b831-4bb4-aa1f-640c65baeeb2.txt` | `stdout/soda_fade_official_seed8_20260518_165700.out` |

`batch_combined_stdout.txt` is the combined remote batch log for seeds 2-8.
`remote_manifest.txt` records the Vast instance metadata for that batch.
