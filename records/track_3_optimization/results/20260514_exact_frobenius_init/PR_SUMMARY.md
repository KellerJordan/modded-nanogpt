# PR Summary: I Exact Frobenius Init

This is a three-seed Track 3 follow-up on top of PR #294's radial damping
optimizer stack. Relative to PR #294, it changes both the `u/w` floor behavior
and the hidden-matrix initialization. The benchmark dataset, batch size, model
architecture, and training endpoint are unchanged.

## What Changed

- Keeps PR #294 outward radial damping and post-step radius correction.
- Changes PR #294's full-update `u/w` floor to the C-style tangent-only `u/w`
  floor: the floor scales only tangent update energy after radial damping,
  leaving the dampened radial component intact.
- Initializes non-projection hidden matrices with
  `std = sqrt(0.33 / fan_in)`, then rescales each matrix to its exact target
  Frobenius norm.
- Affected initialized matrices: `blocks.*.attn.{q,k,v}.weight` and
  `blocks.*.mlp.fc.weight`.
- Projection parameters remain zero-initialized by name, and all non-projection
  biases are zeroed.
- Adds denser near-end validation checkpoints:
  `2960, 2970, 2975, 2980, 2985, 2990, 2995, 2999, 3000, 3010, 3020`.

## Result and Z Score

Three seeds were run: `0`, `1`, and `2`. No seed was dropped.

At step `3010`, the mean validation loss is `3.27748333`:

```text
(3.28 - 3.27748333) * sqrt(3) = 0.00435899 >= 0.004
z = 3.3531
```

At the final step `3020`, the mean validation loss is `3.27683000`:

```text
(3.28 - 3.27683000) * sqrt(3) = 0.00549060
z = 4.2235
```

## Seed-Level Convergence

| Seed | 2960 | 2970 | 2975 | 2980 | 2985 | 2990 | 2995 | 2999 | 3000 | 3010 | 3020 |
| -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| 0 | 3.28025 | 3.27954 | 3.27918 | 3.27885 | 3.27850 | 3.27814 | 3.27782 | 3.27757 | 3.27750 | 3.27684 | 3.27621 |
| 1 | 3.28080 | 3.28008 | 3.27975 | 3.27941 | 3.27903 | 3.27868 | 3.27836 | 3.27813 | 3.27805 | 3.27736 | 3.27670 |
| 2 | 3.28165 | 3.28092 | 3.28058 | 3.28025 | 3.27987 | 3.27952 | 3.27924 | 3.27903 | 3.27895 | 3.27825 | 3.27758 |
| **Mean** | **3.28090000** | **3.28018000** | **3.27983667** | **3.27950333** | **3.27913333** | **3.27878000** | **3.27847333** | **3.27824333** | **3.27816667** | **3.27748333** | **3.27683000** |

Same-seed comparison versus PR #294 logs for seeds `0`, `1`, and `2`
(`PR #294 val - this run val`; positive means this run is lower):

| Step | Seed 0 | Seed 1 | Seed 2 | Mean |
| -: | -: | -: | -: | -: |
| 2875 | +0.00152 | +0.00120 | -0.00146 | +0.00042 |
| 2990 | +0.00149 | +0.00128 | -0.00158 | +0.00040 |
| 2995 | +0.00149 | +0.00131 | -0.00162 | +0.00039 |
| 2999 | +0.00150 | +0.00129 | -0.00164 | +0.00038 |
| 3000 | +0.00150 | +0.00130 | -0.00162 | +0.00039 |
| 3010 | +0.00148 | +0.00130 | -0.00161 | +0.00039 |
| 3020 | +0.00145 | +0.00130 | -0.00158 | +0.00039 |

## Caveat

Because H100 compute was limited (GPU poor), this record only runs three seeds. That is not
enough to claim that exact Frobenius initialization is statistically better than
PR #294's official 11-seed result. The same-seed signal is still useful: seeds
`0` and `1` improve, seed `2` regresses, and the three-seed mean improves by
about `0.00039`, which roughly corresponds to `8-10` training steps using the
Track 3 README's `0.0045` validation-loss-per-100-steps heuristic.

PR #294 clears the Track 3 condition at step `2990` using all 11 reported seeds. The comparison here is
narrower because only seeds `0`, `1`, and `2` were rerun for this experiment; on
that three-seed subset, this run clears the precision condition at step `3010`,
while PR #294's seeds `0`, `1`, and `2` clear it at step `3020`.

## Lineage

Built on PR #294 by @nilin, with inherited components from PR #291, PR #287
PowerCool scheduling, PR #278 SOAP-style preconditioning, PR #275 Contra-Muon,
and PR #274 Skylight/NorMuon-lite.
