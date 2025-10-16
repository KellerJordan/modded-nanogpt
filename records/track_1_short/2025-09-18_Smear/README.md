## New WR 152.7s: Smear token embeddings 1 position forward, -15 steps

This PR builds on all recent WR improvements including PR #127. From inspecting trained model weights, it was observed that multiple attention heads were consistently attending to the prior token. However, attention is a computationally inefficient way to attend to the prior token. This functionality is built-in below in a light-weight manner. The first 12 dimensions of the residual stream/embed are used to gate both the smear module and attention. Approximately, the model finds that (token + 0.07prior_token) is a more useful embedding representation than (token).

Note: This improvement is more marginal than the timing change would indicate. The prior WR had a mean loss of 3.2781. If I attempt to control for loss, the impact of this change appears closer to 5 steps based on testing.

```
self.smear_gate = CastedLinear(12, 1)
self.smear_gate.weight.detach().zero_()

x = self.embed(input_seq)
# smear token embed forward 1 position
smear_lambda = self.scalars[5 * len(self.blocks)]
smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
x = x0 = norm(x[None])
```

Validation:
```
import scipy.stats
import torch

accs = [3.2781,3.2792,3.2765,3.2796,3.2803,3.2801,3.2787,3.2798,3.2787,3.2786]

times = [152.771,152.816,152.834,152.755,152.789,152.773,152.815,152.796,152.798,152.754]

print("p=%.4f" % scipy.stats.ttest_1samp(accs, 3.28, alternative="less").pvalue)
# p=0.0084
print("acc:", torch.std_mean(torch.tensor(accs)))
# acc: (tensor(0.0011), tensor(3.2790))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0269), tensor(152.7901))
```