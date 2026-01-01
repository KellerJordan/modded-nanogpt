## New Record: Bulk Transfer Short Track (-X)

In this PR I bulk transfer the recent short track updates over to the medium track. Instead of porting each feature one by one, I copied the entire train.py file and then made a couple changes from short to medium:
* Layer count 11->16
* model_dim 768 -> 1024
* step count 1845 -> 4600
* cooldown frac 0.5 -> 0.7
* train_max_seq_len 2048 -> 4096
* drop fp8 on lm_head (would take more effort to tune the coef)
* muon lr 0.023 -> 0.015
* adam beta1 0.6 -> 0.8 (wasn't sure if the aggressive 0.6 is appropriate at lower loss regime)
* perform same batch size ramp of (131072, 262144, 393216), then continue and hold at 524288.
* perform same window size ramp of (3, 7, 11, 13), then continue up to 23
* split embed and head at 2/12 of training.
* set backout lambda to 2/3 layers deep
* use the value embed and skip connection patterns from medium track
* Re-add second embedding from #124
* use all 16 attn layers to keep clean split on 8 GPU
* update gates to use 16 dims instead of 12 (same ratio)
* Give attn gate and value embed gate 0.1 lr multiple (YouJiacheng mentioned this was effective on medium track a while back)

Medium track features not yet integrated:
* Snoo and EMA optimizer wrappers
* Smear-MTP from #151
* Sharded mixed precision Muon from record 7.
* Cubic sliding window schedule (I just guessed a ramp that looked ok)
* The logit scaling formula from medium track- I wasn't sure how this was tuned so I just copied the short track formula.

I dropped the lr slightly when I was chasing down a bug where the attention window was larger than the max document length. The remaining parameters above are all first guesses based on my experience on the short track, but I have no scaling intuition yet and many are probably suboptimal. 

## Timing and Validation
```
import scipy.stats
import torch

losses = [3.279, 3.2795, 3.2774, 3.2779, 3.2784, 3.2792, 3.28]
times = [114.508, 114.464, 114.373, 114.343, 114.22, 114.401, 114.369]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
# p=0.0061

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(0.0009), tensor(3.2788))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.0922), tensor(114.3826))
```

timing prior record: 1379.574 
