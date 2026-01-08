## New Record: Even Load of Residual Dims and Value Embeddings (-0.7s)

Updates on PR#190:
- Let gates operate on different chunks of the residual stream
- Reuse 0th value embedding in U-Net pattern

I can retime for PR#191 later.

## Gates

Currently, we have $4$ very distinct gates: smear, skip, attention, and value embedding. For initial embedding $x_t$ at token $t$, a smear gate shifts $x_{t-1}$, and assigns $x_t := x_t + g \times x_{t-1}$. The smear is a cheap proxy for the token shift that attention heads have already been observed to learn at higher cost. A skip gate enables a standard skip connection, and an attention gate controls the standard attention before MLP. Finally, a value embedding adds extra sparse embeddings that are mixed into the value of attention layers.

Each gate serves distinct roles; by the induction heads hypothesis, separate concerns live on separate subspaces of the residual vector. However, the WR had designed the gate to be a function of the first $c = 12$ dimensions, namely
$$
  g(x) = \sigma(w_g \cdot x_{1:c}), \quad w_g: \mathbb{R}^c
$$
which can "crowd" the residual stream with noise. To cleanly separate concerns, we modify gates to operate on distinct chunks of $x$ (with minor overlap).

If gates are "similar" enough in function, we find that some dim overlap can boost the learning signal, especially in early training. We arrived on this configuration below. Although we could grid search the ideal overlap, the main idea is to reduce the pressure on the first 12 dimensions in the residual stream, so this is a reasonable starting point.
```py
smear_dims = (0, 12)
skip_dims = (6, 18)
attn_dims = (18, 30)
ve_dims = (18, 30)
```

## Value Embeddings

After pruning some layers in prior WRs, the placement of the value embeddings became $W_v \in \mathbb{R}^{v \times d}$, for embedding size $d = 768$ and vocabulary $v = 50257$. These embeddings add token-specific information to enrich the block output. We applied this method in a U-Net style over the $11$ Transformer blocks, namely
```
[.,1,2,...,0,1,2]
```
where shared indices share $W_v$. However, we noticed the 0th embedding was only used once, wasting a very large matrix, so we modified the pattern to be symmetric like
```
[0,1,2,...0,1,2]
```
Each embedding matrix is $v \times d = 50257 \times 768 \approx 38.6$ million parameters, and we want to ensure huge parameters are used efficiently.

## Timing

Retiming record of PR#190 at 1840 steps
```py
import scipy.stats
import torch

# timed at 1840 steps

losses = [3.279107093811035, 3.277017831802368, 3.2793996334075928, 3.2764358520507812, 3.2769775390625]
times = [112789.1205839951, 113037.54114699404, 113017.72899700882, 112976.08575701088, 112929.27418399995]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0110

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0014), tensor(3.2778))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(99.0496), tensor(112949.9453))
```

New record timing at 1825 steps
```py

import scipy.stats
import torch

# timed at 1825 steps

losses = [3.2781171798706055, 3.2748520374298096, 3.279273271560669, 3.2784266471862793, 3.2790513038635254, 3.2811760902404785, 3.279404640197754, 3.279543876647949, 3.2793352603912354, 3.2784993648529053, 3.2771427631378174, 3.2776403427124023, 3.278602123260498, 3.276858329772949, 3.2774949073791504, 3.277669668197632, 3.275831937789917, 3.27740216255188, 3.2751529216766357, 3.278644323348999, 3.2792952060699463, 3.280611276626587, 3.277050256729126, 3.2816121578216553, 3.2805702686309814] 
times = [111995.66340900128, 112302.4280989921, 112244.41819799904, 112321.48777899783, 112167.75724300896, 112261.21659600176, 112135.15866100352, 112225.81498399813, 112338.56836100313, 112230.02111700043, 112104.64710800079, 112202.92374401106, 112226.64850599904, 112347.86479799732, 112235.14115900252, 112221.45126000032, 112160.53612099495, 112237.91594400245, 112309.55856299624, 112266.23290100542, 112132.67970199377, 112144.1444540178, 112117.29600099716, 112303.44185298964, 112176.70059901138]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0000

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0017), tensor(3.2784))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(84.2895), tensor(112216.3906))
``` 

If no changes, will merge at latest WR - improvement = 109.2s-(0.7s) = 108.5s to be consistent with 0.7s improvement.