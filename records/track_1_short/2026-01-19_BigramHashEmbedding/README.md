## New Record: Bigram Hash Embedding (-5.6s, -165 steps)

Updates in PR:
* Bigram Hash Embedding (-5.1s)
* Fix partial key offset to apply to stationary dims instead of 50/50 stationary/rotating (-0.5s) [bug was introduced in paired head PR]
* Reduce step count from 1765 to 1600
* Increase cooldown frac from 0.5 to 0.55 (I find it works to increase this as step count decreases)

Bigram hash code which runs on CPU during each dataloader iteration:
```
args.bigram_vocab_size = 5 * vocab_size
def get_bigram_hash(x):
    """
    Computes bigram hash for each position using [prev_token, curr_token].
    Multiply by arbitary large ints to get even spread over int32 range.
    Position 0 is mapped to the reserved index (vocab_size - 1).
    BOS_tokens within the batch will hash based on last token of prior doc. Masking this ran slower and showed no improvement.
    """
    rand_int_1 = 36313
    rand_int_2 = 27191
    mod = args.bigram_vocab_size-1
    x = x.to(torch.int32).clone()
    x[0] = mod
    x[1:] = torch.bitwise_xor(rand_int_1 * x[1:], rand_int_2 * x[:-1]) % mod
    return x
```

Model structure:
```
# model args
embed = nn.Embedding(vocab_size, model_dim)
bigram_embed = nn.Embedding(bigram_vocab_size, model_dim).zero_init()
x0_lambdas = nn.Parameter(torch.zeros(num_layers))
bigram_lambdas = nn.Parameter(0.1*torch.ones(num_layers))

# model forward pass
x = x0 = norm(nn.Embedding(input))
x0_bigram = nn.Embedding(get_bigram_hash(input))
for i in range(num_layers):
    x = x + x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram
    x = block(x)
logits = lm_head(x)
```

Recent discussion on Deepseek's Engram got me looking into hash embeddings, and this 2017 paper: https://arxiv.org/abs/1709.03933. Meaningful hash collisions are rare due to token distribution, so hashing seems good. 

I did some of testing on more advanced ways to integrate the hashed embeddings with gating, trigrams, and multiple hashes. Very simple direct addition to the residual stream outperformed by a decent margin.

### Ideas
Interestingly, the model now has more parameters than training tokens. Typically when a feature gets added the step count is reduced. At some point it may be better to drop an attn/mlp.

Now that we have an additional 5*50304 params getting communicated across GPUs and the bottleneck is overwhelmingly on comms, it could be worthwhile to build a sparse communication approach to only send embeds with nonzero grads. I did not look at the profiler here or try other comms orderings. Due to stepping Adam every other step, early step times alternate between 31ms and 37ms. (Could be other ideas like step bigram embed once every 4 steps)

There is an average of 0.4s extra due to a stall on step 6. Chris pointed this out and I believe Varun has been working on a fix. I may have also gotten a worse machine here that was making the issue more pronounced.

## Timing and Validation
```
import scipy.stats
import torch

losses = [3.2756, 3.2786, 3.2773, 3.2791, 3.2778, 3.277]
times = [98.065, 98.017, 99.201, 99.151, 98.112, 98.031]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0025

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0012), tensor(3.2776))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.5794), tensor(98.4295))
```

retiming prior record: 104.062 [104.129, 103.995]

If no changes, will merge at 99.3s to be consistent with 5.6s improvement over 104.9s.
