# Tuned-NanoGPT

This is a variant of the [Python GPT-2 trainer](https://github.com/karpathy/llm.c/blob/master/train_gpt2.py) from
Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) repo. It trains to the same quality 35% faster than the original.

To run:
```
python data/fineweb.py
./run.sh
```

It yields 3.2798 perplexity on the Fineweb validation set after training a 124M-parameter transformer for 6.44B tokens.

For comparison, the llm.c trainer yields 3.2847 perplexity on the same after training for 10B tokens. (1.55x more)

The speedup of this trainer over the original are due to the following changes:
- Increased learning rate
- Halved batch size (but ~same training speed)
- Improved learning rate schedule (we use a 256-step linear rampup, then a linear rampdown to 0.1 * lr_max)
- Normalized the gradient for each weight to have unit norm
- Removed all affine scale and bias parameters from the architecture, and switched to RMSNorm (actually this causes a slight slowdown, and I just did it to reduce code complexity)

