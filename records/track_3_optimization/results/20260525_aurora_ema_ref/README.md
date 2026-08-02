# Aurora EMA Reference 2860 Results

This result uses the Aurora/EMA-Nesterov base from PR #309 with a fixed
reference-interpolated output rule. It captures reference weights at step 2375,
uses `gamma=-0.075` from step 2850 onward, and reports the fixed stopping point
K=2860.

The archived script hardcodes the submitted hyperparameter defaults. It reads
`--seed` to select the random seed for a reproducibility run; other command-line
arguments only control logging.

## Configuration

| field | value |
|---|---|
| train steps | 2900 |
| fixed stopping point K | 2860 |
| schedule steps | 2980 |
| LR power | 1.2 |
| base method | Aurora/EMA-Nesterov from PR #309 |
| CGI alpha | 0.0 |
| zero non-projection biases | true |
| reference capture step | 2375 |
| reference apply start step | 2850 |
| reference gamma | -0.075 |
| EMA-Nesterov stepsize | 0.3 |
| EMA-Nesterov lookahead EMA | 0.99 |
| EMA-Nesterov prefill steps | 300 |
| EMA-Nesterov rest step | 1950 |

## Reliability

Across 16 non-cherry-picked seeds at the fixed stopping point K=2860, the mean
validation loss is 3.278939375. The Track 3 significance rule is:

`(3.28 - mean) * sqrt(n) >= 0.004`

For these runs:

`(3.28 - 3.278939375) * sqrt(16) = 0.00424250 >= 0.004`

| seed | val loss at K=2860 | logfile |
|---:|---:|---|
| 0 | 3.27765 | `c4b5e2ad-f59c-4809-86ff-913338ae7a2c.txt` |
| 1 | 3.27965 | `d715cb87-fbdf-44cb-8798-b505f8e8b71a.txt` |
| 2 | 3.27675 | `7b578310-381e-469b-94f7-8bfe3dd47bb9.txt` |
| 3 | 3.27978 | `6f0d30bb-483d-4803-9460-153a0071250d.txt` |
| 4 | 3.27961 | `58808330-3c1f-4d8b-8108-2f62a646063f.txt` |
| 5 | 3.27999 | `bb2723d5-cf02-440a-ad69-cdc5dbe6c251.txt` |
| 6 | 3.27696 | `cb2d26ac-e884-44e4-98b1-b4a15aafd415.txt` |
| 7 | 3.27746 | `242f921c-734e-4617-b361-7978de6c12d4.txt` |
| 8 | 3.27929 | `5c30823b-1631-41f6-901b-7d2bb3f1b53b.txt` |
| 9 | 3.27910 | `e782d1d7-8f24-4373-b443-5fd33a56995b.txt` |
| 10 | 3.28025 | `a5ad12b6-e74e-4612-b76b-66db831fc95f.txt` |
| 11 | 3.28060 | `73926855-fa53-46d9-aa5e-0797b6550f93.txt` |
| 12 | 3.27822 | `de02a52e-eeac-41e8-85e2-387b2d960e2b.txt` |
| 13 | 3.27837 | `15223178-b211-42d7-a6ba-b969c3148973.txt` |
| 14 | 3.27969 | `9df54a64-b0d7-4132-9da6-32ec2caad7a2.txt` |
| 15 | 3.27966 | `8b1ae377-e835-4c90-a6b8-75afaf085b04.txt` |
| **Mean** | **3.278939375** | |

All logs report `step:2860/2900`; no per-run early stopping is used.

## Reproduction

Run one seed from the repo root with the Track 3 quickstart environment:

```bash
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) records/track_3_optimization/results/20260525_aurora_ema_ref/train_gpt_simple_aurora_ema_ref.py --seed 0
```
