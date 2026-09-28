# MaxEntSlop momentum and rate-following decay on the tuned Muon baseline, 3060 steps

Eight consecutive seeds (0–7) trained on the 3060-step schedule with `train_gpt_muon_maxentslop.py`; per-seed final validation losses in `summary.tsv`, full file logs (embedded source and environment) in `seed0.txt … seed7.txt`. Hardware: Lambda 8×A100-80GB, PyTorch 2.11.0+cu128. `momentum_kernels.gif` shows the momentum kernels over training; `val_loss_8seed.png` the validation loss of the eight seeds against the K-Maxwell (PR #357) and bi-Maxwell (PR #340) records.

```bash
for seed in 0 1 2 3 4 5 6 7; do
  torchrun --standalone --nproc_per_node=8 train_gpt_muon_maxentslop.py --seed "$seed" --train_steps 3060 || exit
done
```
