## New WR 1.25% better than PR #122: Optimize distributed training, improve skip connection gating, and enhance bfloat16 usage

This PR takes all recent improvements including PR #122 from today, and adds on top of that the following three ideas:

- Replacing in the Muon optimizer Python for-loops with vectorized tensor operations using PyTorch. This is done for improved gradient sharding, padding, and parameter synchronization.

- Cast more tensors and buffers (embeddings, linear layers, optimizer state, positional encodings) to torch.bfloat16. This allows us to get faster experiments with minimal changes in model accuracy.

- Apply sigmoid gating to U-Net skip connections; initialize skip weights to -1.5 for better learnability. Instead of directly multiplying skip connections by a raw trainable parameter (which could be unbounded and unstable), the code now passes the skip weight through a sigmoid function. This constrains the gate value to the range (0, 1), making the effect of each skip connection smoothly adjustable and numerically stable.

This improves the runtime by 2 seconds, i.e. 1.25%, see below.

### Validation
I’ve used a 8 × H100 SXM NVLink 80GB node on RunPod. The results I’ve been getting when benchmarking PR #122 are a bit better than the ones reported there. So here I present the statistics of both PR #122 and this PR when using that node:


Validation for PR #122
```
import scipy.stats
import torch

accs = [3.2798, 3.2798, 3.2829, 3.2785, 3.2783, 3.2787, 3.2787, 3.2784, 3.2821, 3.2794, 3.2786, 3.2765, 3.2794, 3.2776, 3.2778, 3.2774, 3.2777]

times = [157.977, 157.889, 158.014, 158.103, 158.093, 158.001, 158.089, 157.981, 158.019, 157.963, 158.043, 157.957, 157.880, 157.687, 158.002, 157.947, 158.097]

print("p=%.4f" % scipy.stats.ttest_1samp(accs, 3.28, alternative="less").pvalue)
# p=0.0069

print("acc:", torch.std_mean(torch.tensor(accs)))
# acc: (tensor(0.0016), tensor(3.2789))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.1021), tensor(157.9848))
```

Validation for PR #123
```
import scipy.stats
import torch

accs = [3.277, 3.2772, 3.2778, 3.2767, 3.2805, 3.2781, 3.2797, 3.2802, 3.2774, 3.2767, 3.2769, 3.2783]

times = [155.902, 155.956, 156.043, 155.987, 155.980, 155.717, 156.019, 156.077, 156.064, 156.100, 156.129, 155.799]

print("p=%.4f" % scipy.stats.ttest_1samp(accs, 3.28, alternative="less").pvalue)
# p=0.0002

print("acc:", torch.std_mean(torch.tensor(accs)))
# acc: (tensor(0.0014), tensor(3.2780))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.1233), tensor(155.9811))
```