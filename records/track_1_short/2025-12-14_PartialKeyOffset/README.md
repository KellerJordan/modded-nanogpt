## New WR 128.8s: Partial Key Offset

Five updates, for roughly 2.4s improvement:
1. Implement a partial key offset for the long sliding windows
2. Merge the `x_lambda*x + x0_lambda*x0` for layer 0 into (x_lambda+x0_lambda)*x. And clean up the code to properly represent the 11 layer model.
3. Drop 50 steps. (60ms per step)
4. Align the batch size schedule to update exactly when the window size updates (0.33!=1/3)
5. Zero out the initial value embeddings. Very minor impact, but may give lower variance and prefer zero init as lowest assumption config.

The partial key offset is only applied to the stationary head dims (32-64 and 96-128). This was found to perform better than applying it to all dims. This approach gives the queries more freedom to attend to multiple positions at once through a single key. I tested applying to only a subset of heads and got worse results. The lowest loss was achieved when applying it to every layer, but the cost:speed ratio seems best when only applying to the long windows, which are the ones primarily responsible for induction.

```
# shift keys forward for the stationary head dims. Enables 1-layer induction.
k[:, 1:, :, self.head_dim//4:self.head_dim//2] = k[:, :-1, :, self.head_dim//4:self.head_dim//2]
k[:, 1:, :, self.head_dim//4+self.head_dim//2:] = k[:, :-1, :, self.head_dim//4+self.head_dim//2:]
```

<img width="3471" height="2388" alt="Picture4" src="https://github.com/user-attachments/assets/bd861f9e-4fed-47f0-b9fd-ee14f692efa9" />


## Timing and Validation
```
import scipy.stats
import torch

losses = [3.2788,3.2774,3.2786,3.2792,3.2762,3.2769,3.2781,3.2778,3.2761,3.2783,3.2809]
times = [128.892,128.907,128.912,128.844,128.822,128.869,128.818,128.882,128.886,128.95,128.946]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0004

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0014), tensor(3.2780))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0443), tensor(128.8844))
```

retiming prior record: 131.2: [131.270,131.241,131.213]
(appears I got a slightly slower machine this time)
