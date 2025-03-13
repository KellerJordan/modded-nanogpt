# Adding HellaSwag validation

This run includes running validation on 10042 HellaSwag validation tasks (set `hellaswag=True`). After 1770 steps (0.6 cooldown_frac) the HellaSwag accuracy is `0.2913` (validation loss `3.2779`).

When running with different steps/cooldown_frac, we saw the following HellaSwag accuracies:
- `0.3103` (4000 steps, 0.7 cooldown_frac)
- `0.3295` (10000 steps, 0.95 cooldown_frac)

Time to process HellaSwag:
- First time we run HellaSwag validation inside the script: ~30s. Thereof ~15s to download and preprocess the data and ~15s (includes warm-up) to run forward passes and evaluate
- Each subsequent validation on 10042 HellaSwag tasks inside the same script: ~0.4s (~0.2s per sequence times 2 sequences per GPU)
