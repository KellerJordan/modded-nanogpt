Paired-Gram ANVIL research
=========================

The experiment re-associates two ANVIL quintic maps in the smaller Gram matrix,
then applies their combined affine transform to the rectangular update once.
It recomputes the Gram between pairs and retains scalar identity coefficients
outside BF16 diagonal storage. The algebra is tested in float64.

Accounting for the trainer's mirrored symmetric kernels, the ideal per-pair
cost changes from `6 M k² + 2 k³` to `3 M k² + 8 k³`, if both polynomial squares
use symmetric kernels. That suggests 15–21% fewer arithmetic operations for the
current rectangular bank shapes. This is an arithmetic estimate; launches,
buffer traffic and kernel efficiency determine actual runtime.

The general Gram-space strategy has prior art in [Dion3](https://arxiv.org/abs/2608.11612)
and [Polar Switch](https://arxiv.org/abs/2610.26742). This is an ANVIL-specific
paired-map experiment within that line of work.

```bash
python -m experiments.anvil_gram_pairs --output work/gram-pairs.json
```

The BF16 diagnostic deliberately includes rank-one and rotated ill-conditioned
matrices. It exposes material drift even where float64 re-association agrees.
The candidate is **not enabled in the trainer**, has no CUDA timing, and has not
established acceptable training loss. It needs rounding-error compensation or a
better-conditioned circuit before it is a defensible optimizer replacement.
