# Tuned-NanoGPT

This is a variant of the [PyTorch GPT-2 trainer](https://github.com/karpathy/llm.c/blob/master/train_gpt2.py) from
Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) repo. It both:
* Trains 35% more efficiently (6.44B tokens instead of 10B to reach the same validation loss).
* Has simpler code (414 lines instead of 858).

To simplify the code, some features were removed, like text generation. And to obtain a training speed improvement, we have diverged
a bit architecturally and in terms of hyperparameters from being a strict reproduction of the GPT-2 paper.

To run it:
```
python data/fineweb.py
./run.sh
```

This will produce a 124M-parameter transformer trained on 6.44B tokens, which has has 3.2753 validation loss on the Fineweb validation set.

For comparison, the original llm.c trainer yields 3.2847 validation loss after training for 10B tokens.

This speedup is due to the following changes:
- Increased learning rate by 3x (this is the main thing)
- Improved learning rate schedule (a 256-step linear rampup, then a linear rampdown to 0.1 * lr_max)
- Normalized the gradient for each parameter to have unit norm
- Removed all affine scale and bias parameters from the architecture, and switched to RMSNorm (actually this causes a slight slowdown, and I just did it to reduce code complexity)
- Removed the special initialization for linear layers before residuals. Instead, just scale down the output of the attention block by a fixed scalar.

Note: running this trainer for the full 10B tokens yields a validation loss of 3.2267.

