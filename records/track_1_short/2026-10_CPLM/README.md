# Copy-sink pointer LM (CPLM) on record #92

**To fill after the 8xH100 runs:** record time, losses, p-value, logs.

## Change

The next-token distribution becomes a mixture of the LM softmax and a pointer over the document's previous tokens:

    p(y) = p_lm(y) * (1 + alpha * a_sink / (1 - alpha)) + alpha * p_copy(y)

- `alpha` is the softmax mass of a `<copy>` slot: vocab id 50257, one of lm_head's padding rows, under the same
  softcap as the other logits. No new output parameters.
- `p_copy` is a single-head pointer (d=128) over the final hidden state: document-masked attention over the
  previous tokens (2048-token band in training) plus a learned sink key; `a_sink` is the sink's mass, which is
  returned to the LM. Queries and keys are RMS-normed, with a learnable query gain (QK-norm), which removes the
  logit blow-up that otherwise kills the pointer on some seeds.
- Training: the mixture runs inside #92's fused fp8 cross-entropy kernel (`perf/kernels/cplm_cross_entropy.py`);
  the pointer is a fused flash-style Triton kernel (`cplm_copy.py`). Extra parameters: 2x128x768 + 129 (~0.2M).
- Validation: same mixture on the full-softmax, canonical-masked logits (`<copy>` exempt from the mask), with an
  8192-token pointer band.
- Steps: 1194 -> 1035 (`NUM_SCHEDULED_ITERATIONS` 1122 -> 963). Everything else is #92 unchanged.

Settings are defaults in `train_gpt.py` (env-overridable): `CPLM=1 CPLM_QK_NORM=1 NUM_SCHEDULED_ITERATIONS=963
CPLM_EVAL_BLOCK=8192`. `CPLM=0 NUM_SCHEDULED_ITERATIONS=1122` runs the #92 baseline from the same code.

## Run

    ./run.sh                                   # = torchrun --standalone --nproc_per_node=8 train_gpt.py
    python records/track_1_short/2026-10_CPLM/summarize.py logs/*.txt

## Preliminary results (4xGH200, same-node A/B vs #92)

- 1035 steps, 8 seeds: mean val loss 3.2775 (p < 0.001).
- Per-step cost of CPLM vs #92: +1.3% (copy kernel, projection, extra fused norms); total ~ -12% training time.
