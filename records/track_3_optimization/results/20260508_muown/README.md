# Muown — Track 3 Submission

Submission authored by: fhuebler, kcc-lion

Muown decomposes every 2-D weight matrix into a per-row gain `g` and a
direction vector `v` via weight normalization (`W = g * v / ||v||`), then
updates the two factors with different geometries:

* **Direction (`v`):** Nesterov-style momentum followed by Newton-Schulz
  orthogonalization (Polar Express, 12 iterations) of the update.  The
  orthogonalized update is applied to `v` with a scale proportional to
  `sqrt(max(m, n))`, gated by `direction_scale=0.2`.
* **Gain (`g`):** Standard Adam with bias correction (`betas=(0.9, 0.95)`).
* **V-norm schedule (angular step-size attenuation):** After each step, the
  stored `||v||` is smoothstep-interpolated from its initial value toward a
  fixed `v_norm_target_scale`.  Because the weight-norm recomposition
  normalizes `v` before multiplying by `g`, growing `||v||` relative to the
  actual update magnitude shrinks the effective angular displacement on the
  unit sphere, attenuating the angular step size as training progresses.
  The target scale is layer-family-dependent (`mlp.proj` receives a 3.33x
  multiplier; all other families use 1x).

Auxiliary parameters (embeddings, head, 1-D norms/biases) are trained with
fused AdamW at per-group learning rates.

## Key hyperparameters

| Parameter | Value |
| - | - |
| `lr_muown` | 0.0035 |
| `momentum` | 0.95 |
| `ns_steps` | 12 |
| `direction_scale` | 0.2 |
| `v_norm_target_scale` | 2.688 (x family mult) |
| `lr_cooldown_frac` (both) | 0.7 |
| `lr_embed` / `lr_head` / `lr_1d` | 0.9 / 0.009375 / 0.03 |
| `batch_size` | 524288 |
| `train_steps` | 3200 |

## Statistical significance

The benchmark requires `(3.28 - mu) * sqrt(n) >= 0.004`.  For n=10, this yields a threshold of
`mu < 3.28 - 0.004/sqrt(10) = 3.27874`.

Ten non-cherry-picked seeds (1-10) were run for 3200 steps.  The
significance condition is first satisfied at **step 3125**:

| Step | Mean | Std | Min | Max | (3.28-mu)*sqrt(10) |
| -: | -: | -: | -: | -: | -: |
| 3100 | 3.27877 | 0.00080 | 3.27754 | 3.27979 | 0.00390 |
| **3125** | **3.27752** | **0.00080** | **3.27636** | **3.27859** | **0.00784** |
| 3150 | 3.27641 | 0.00081 | 3.27525 | 3.27748 | 0.01135 |
| 3175 | 3.27569 | 0.00081 | 3.27451 | 3.27676 | 0.01362 |
| 3200 | 3.27534 | 0.00081 | 3.27415 | 3.27642 | 0.01474 |

At step **3125**, `(3.28 - 3.27752) * sqrt(10) = 0.00784 >= 0.004`.

## Per-seed final losses (step 3200)

| Seed | Val loss |
| -: | -: |
| 1 | 3.27541 |
| 2 | 3.27642 |
| 3 | 3.27600 |
| 4 | 3.27415 |
| 5 | 3.27607 |
| 6 | 3.27569 |
| 7 | 3.27558 |
| 8 | 3.27548 |
| 9 | 3.27421 |
| 10 | 3.27437 |
| **Mean** | **3.27534** |
