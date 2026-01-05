## New Record: Mixed Precision Interweaved Optimizer (-1.0s)

Updates in PR:
* Update MLP and ATTN weights from float32 to bfloat16
* Add a mantissa buffer to Muon such that parameter updates can be tracked in float32 even though weights are in bfloat16. Following work of YouJiacheng on Medium track record 7.
* Enable Adam beta values to be specified on a module level, and slightly adjust these
* Interleave Muon with Adam. Heavily based on the ideas of Chris McCormick here: https://github.com/KellerJordan/modded-nanogpt/discussions/170.
* Add 15 steps (not strictly necessary based on loss)

Without the mantissa tracking the bfloat16 update is substantially detrimental to loss. 

I observed that the lm_head/embed slightly benefits from a very low beta1 term in Adam (0.5), whereas standard convention is closer to 0.9 for Adam parameters. I hypothesis that this is driven by the asymmetrical gradient distribution caused by the nature of low frequency tokens. Consider a parameter with a very large gradient signal once every 100 steps. A small beta1 causes the parameter to take a massive step in the direction of the large gradient, and then quickly fall back to whatever the weak gradient signal is when the parameter is not the target. A large beta1 causes the parameter to continue moving in whatever direction activated it for 100+ steps, even after the signal is long gone and the network has a new topology. It may be worth exploring different beta1 values for different tokens as a function of token frequency. I find it quite unlikely that a token that occurs once every 1 million tokens should have the same beta as a token that occurs once every 100 tokens.

Muon.step() is separated into 3 modular components, such that when Adam.step() is called Adam can selectively interweave Muon components. 
* Adam first immediately calls the muon reduce scatter
* Adam kicks off every all gather except for the last value embed
* muon runs polar express and kicks off its all gathers and completes its first all gather and parameter copy
* Adam kicks off its last all gather
* Muon runs its final parameter copy
* Adam finishes its last all gather.

This enables close to zero communication downtime.

<img width="1434" height="377" alt="image" src="https://github.com/user-attachments/assets/05eee60f-ffd4-4327-b437-47310d121c17" />

Whereas before:
<img width="998" height="310" alt="image" src="https://github.com/user-attachments/assets/85a0b4a9-0ad6-4933-a669-1d0f231e3674" />


## Timing and Validation
```
import scipy.stats
import torch

losses = [3.2795, 3.2789, 3.2776, 3.2763, 3.2785, 3.278, 3.2774, 3.2777, 3.2802, 3.2767]
times = [112.774,112.798,112.837,112.767,112.805,112.802,112.822,112.755,112.932,112.953]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0004

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0012), tensor(3.2781))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0671), tensor(112.8245))
```

retiming prior record: 113.895 [113.818, 113.896, 113.97]

If no changes, will merge at 112.7s to be consistent with 1.0s improvement.
