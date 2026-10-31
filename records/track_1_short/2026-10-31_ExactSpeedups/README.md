# Track 1: exact speedups on top of #393 (30.67 s, n=12; −0.66 s vs #393 on the same node)

The code change is in the companion PR: four overheads removed from #393's training step. They are on by default, and `EXACT_LEVERS=none` restores #393. This folder holds the runs.

## Result

All runs are on one 8×H100 node: RunPod EUR-NO-2, H100 80GB HBM3 SXM, NV18 between all pairs, driver 595.91.07, torch 2.10.0+cu128. #393 (stock) was timed alongside on the same node.

| | n | train_time | val_loss |
|---|---|---|---|
| **this PR** | 12 | **30.6746 s ± 0.0169** | **3.27743 ± 0.00116** |
| #393, same node | 6 | 31.3232 s ± 0.0324 | 3.27705 |

- **Paired (5 seed pairs, alternating order):** -0.659 s ± 0.047 (t = 31.1).
- **Loss:** one-sided t-test of mean val_loss < 3.28 at n = 12: t = 7.66. p < 0.01 needs t > 2.718, so it passes.
- **Run order:** the 12 runs of this PR were declared before running (seeds 3–7 paired with #393, 101–107 alone) and every run counts. Seeds 106 and 107 were added to reach n = 12, the same n as #393. That was proposed before the seed-101 run started and approved while the extra runs were going.

### Every run

| run | arm | seed | train_time (s) | val_loss | log |
|---|---|---|---|---|---|
| X12 | this PR | 3 | 30.660 | 3.2785 | `this_pr/3012f81e-feb3-4e6e-8a65-d21fc088d0e5.txt` |
| X13 | this PR | 4 | 30.666 | 3.2780 | `this_pr/c7b7de44-4f31-4183-adbc-8dd0c0b4d73c.txt` |
| X16 | this PR | 5 | 30.636 | 3.2771 | `this_pr/73cf8cc3-0ef1-4184-a166-20aa55763bf4.txt` |
| X17 | this PR | 6 | 30.675 | 3.2766 | `this_pr/5b1005e6-acb2-4286-8d1b-2ff8e8dbf403.txt` |
| X20 | this PR | 7 | 30.679 | 3.2761 | `this_pr/77a7f8bd-9b58-445e-ae5a-9cf5d7d2970c.txt` |
| X21 | this PR | 101 | 30.661 | 3.2768 | `this_pr/90f71839-6dac-4d79-abd9-411fe31df20f.txt` |
| X22 | this PR | 102 | 30.677 | 3.2789 | `this_pr/46d4b5f0-b11f-4804-9749-ddf3bb0796ab.txt` |
| X23 | this PR | 103 | 30.675 | 3.2789 | `this_pr/e56b0c1b-0562-4128-9c99-ddbef1568679.txt` |
| X24 | this PR | 104 | 30.694 | 3.2754 | `this_pr/b08833b0-e223-448d-96cd-a0a80d93a918.txt` |
| X25 | this PR | 105 | 30.687 | 3.2767 | `this_pr/4ae9a5e5-85a2-41f7-ab12-b3339637cc75.txt` |
| X26 | this PR | 106 | 30.693 | 3.2776 | `this_pr/61cc1b88-68e3-4bec-b65b-8da7fb231abf.txt` |
| X27 | this PR | 107 | 30.692 | 3.2786 | `this_pr/00d08fc1-e9b3-4445-b758-052eba5aad07.txt` |
| X01 | #393 | 1 | 31.326 | 3.2756 | `baseline_393/82cfb997-212e-477e-96bc-33ab95855031.txt` |
| X11 | #393 | 3 | 31.307 | 3.2786 | `baseline_393/e288f48d-75e2-4026-9a19-6c78535bf10d.txt` |
| X14 | #393 | 4 | 31.275 | 3.2789 | `baseline_393/9feb84e5-3ac0-4869-a913-ce437095154a.txt` |
| X15 | #393 | 5 | 31.374 | 3.2764 | `baseline_393/3870fa47-51ca-4d51-83a5-87ebf84d2cd1.txt` |
| X18 | #393 | 6 | 31.325 | 3.2771 | `baseline_393/17c707d4-b2cc-4271-909f-cf6fc09c9928.txt` |
| X19 | #393 | 7 | 31.332 | 3.2757 | `baseline_393/7cfb34a8-9578-4153-9732-92806b7eac91.txt` |

X01 (#393, seed 1) was the node's first run and includes a cold compile; that adds ≈ 0.02 s.

## Paired by seed

| seed | #393 | this PR | Δ |
|---|---|---|---|
| 3 | 31.307 s | 30.660 s | -0.647 s |
| 4 | 31.275 s | 30.666 s | -0.609 s |
| 5 | 31.374 s | 30.636 s | -0.738 s |
| 6 | 31.325 s | 30.675 s | -0.650 s |
| 7 | 31.332 s | 30.679 s | -0.653 s |

## Per lever (one run each vs #393, seed 1)
| lever | Δ |
|---|---|
| `fa3copy` | −0.477 s |
| `rsearly` | −0.077 s |
| `bankinplace` | −0.021 s |
| `prescatter` | −0.017 s |

## How to run
Same as #393; the levers are on by default.
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Disclosures
- **Hardware:** RunPod, not PrimeIntellect. Stock #393 took 31.32 s on this node (31.285 s reported in #393), so the claim is the same-node difference.
- **Timed tree:**
  - The runs used exactly this PR's lever code, switched on with `EXACT_LEVERS=all`.
  - The tree also carried my launch-kit's off-by-default timing hooks (phase stamps and markers), active in both arms alike, so the code printed at the top of each log includes them.
  - The PR itself leaves them out, and I did not re-time the stripped tree.
- **Loss:** unchanged within run-to-run noise. #393's runs are not bit-reproducible either, because FA3's default backward and two atomic adds are non-deterministic. In a deterministic check on 1×H100 with #393's per-rank shapes, losses, all weights and 24 of 25 gradients were bitwise identical. The exception is #393's own atomic scatter-add, which differs the same way between two #393 runs.

## Taken from other PRs
Everything except the four levers is #393 by @francesco-prignano.

