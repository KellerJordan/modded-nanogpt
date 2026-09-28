# MaxEntSlop momentum on MuonH fast-slow decay, 3010 steps

Eight consecutive seeds (0–7) trained on the 3010-step schedule with `train_gpt_muonh_maxentslop.py` and `--min_lr 0.0005018182` (the MuonH learning rate ends at that value instead of zero; see the PR). The script also sets, before Newton–Schulz, the component of the momentum sum parallel to the current weight matrix to 0.044 times the norm of its perpendicular component, sign kept (`RADIAL_TO_TANGENT_RATIO` in the script). Per-seed final validation losses in `summary.tsv`, full file logs (embedded source and environment) in `seed0.txt … seed7.txt`. Hardware: Lambda 8×A100-80GB, PyTorch 2.11.0+cu128. `momentum_kernels.gif` shows the momentum kernels over training; `val_loss_8seed.png` the validation loss of the eight seeds against the K-Maxwell record (PR #359).

```bash
for seed in 0 1 2 3 4 5 6 7; do
  torchrun --standalone --nproc_per_node=8 train_gpt_muonh_maxentslop.py --seed "$seed" --train_steps 3010 --min_lr 0.0005018182 || exit
done
```
