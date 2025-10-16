## New WR 140.7s: Backout, Misc Hyperparam cleanup, Fix Lambda Count

This submission reflects all recent WR changes up to [PR#136](https://github.com/KellerJordan/modded-nanogpt/pull/134).

1. Implementing #138 by @snimu with some tuning for the short track. In standard transformer architecture contributions to the residual stream have to serve two purposes at once: provide context to downstream layers, and add to the final prediction. Information may be valuable for downstream context but not directly useful for the final prediction. A lambda is added such that context added to the residual stream in the first 8 layers can be backed out before the final prediction.
2. Hyperparam tuning, pulling down number of steps following last PR
    * num_iterations: 2380->2290
    * cooldown_frac: 0.4->0.45
    * adam beta1: 0.7->0.65
3. Cleanup extra lambda params, dropping count from 72 to 64. This fixes the hiccups/stuttering and saves 1.8s.

Layer 8 was chosen after implementing a lambda per_layer version and observing these coefficients:
[0.5400, 0.4613, 0.4364, 0.3429, 0.2675, 0.3030, 0.2023,0.3761,-0.0741,  -0.2164,  -0.2905]
```
    if i==8:
        x_backout=x

# back out contributions from first 8 layers that are only required for downstream context and not direct prediction
x -= backout_lambda*x_backout
x = norm(x)
logits = self.lm_head(x)
```

Dropping the extra torch.zeros(num_layers), brings scalars to 64 instead of 72, for a clean 8 params per GPU instead of 9.
```
pad = (-num_layers * 5 - 2) % dist.get_world_size()
self.scalars = nn.Parameter(
    torch.cat(
        [
            -1.5
            * torch.ones(num_layers),  # skip_weights -> σ(-1.5) ≈ 0.18
            *[
                torch.tensor([1.0, 0.0]) for _ in range(num_layers)
            ],  # block lambdas
            *[
                torch.tensor([0.5, 0.5]) for _ in range(num_layers)
            ],  # SA lambdas
            torch.zeros(1), # smear_lambda
            0.5*torch.ones(1), # backout_lambda
            torch.ones(pad),
        ]
    )
)
```
If I try to drop this down to 56 by removing 6 extra skips and 2 padding lambdas, the runtime goes up slightly. It appears that param size of 64 performs better in Adam than 56 or 72. Since Adam splits the param 8 ways across GPUs, this means Adam is performing better with an array of size 8 per GPU instead of 7 or 9.

## Timing and Validation
```
import scipy.stats
import torch

losses = [3.2772, 3.2796, 3.2781, 3.2783, 3.2769]
times = [140.626, 140.678, 140.693, 140.718, 140.769]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0070

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0011), tensor(3.2780))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0525), tensor(140.6968))
```

retiming prior record: 146.9: [147.189,146.906,146.690]
