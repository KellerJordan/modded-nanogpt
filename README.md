# Tuned-NanoGPT

This is a variant of the [Python GPT-2 trainer](https://github.com/karpathy/llm.c/blob/master/train_gpt2.py) from
Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) repo. It both:
* Trains 35% more efficiently than the original.
* Has simpler code: 419 lines instead of 858.

To run it:
```
python data/fineweb.py
./run.sh
```

This will produce a 124M-parameter transformer trained on 6.44B tokens, which has has 3.2798 validation loss on the Fineweb validation set.

For comparison, the original llm.c trainer yields 3.2847 perplexity after training for 10B tokens. (1.55x more tokens)

This speedup is due to the following changes:
- Increased learning rate by 2.5x (this is the main thing)
- Halved batch size (but ~same training speed)
- Improved learning rate schedule (a 256-step linear rampup, then a linear rampdown to 0.1 * lr_max)
- Normalized the gradient for each parameter to have unit norm
- Removed all affine scale and bias parameters from the architecture, and switched to RMSNorm (actually this causes a slight slowdown, and I just did it to reduce code complexity)
- Removed the special initialization for linear layers before residuals. Instead, just scale down the output of the attention block by a fixed scalar.

Note: running this trainer for the full 10B tokens yields a validation loss of 3.2267.

