# Adding HellaSwag validation

This run includes running validation on 10042 HellaSwag validation tasks (set `hellaswag=True`). After 1770 steps (and 0.4 cooldown_frac) the HellaSwag accuracy is `0.2945` and validation loss is `3.2797`.

On a different run with 4000 steps (and 0.7 cooldown_frac) we got HellaSwag accuracy `0.3160` and validation loss `3.1568`.

Time to process HellaSwag:
- First time we run HellaSwag validation inside the script: ~30s. Thereof ~15s to download and preprocess the data and ~15s (includes warm-up) to run forward passes and evaluate
- Each subsequent validation on 10042 HellaSwag tasks inside the same script: ~0.4s (~0.2s per sequence times 2 sequences per GPU)
