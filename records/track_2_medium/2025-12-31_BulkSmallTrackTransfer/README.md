## New Record: Bulk Transfer Short Track (-5.63 Minutes)

In this PR I bulk transfer the recent short track updates over to the medium track. Instead of porting each feature one by one, I copied the entire train.py file and then made a couple changes from short to medium:
* Layer count 11->16
* model_dim 768 -> 1024
* step count 1845 -> 4740
* cooldown frac 0.5 -> 0.7
* train_max_seq_len 2048 -> 4096
* drop fp8 on lm_head (would take more effort to tune the coef)
* muon lr 0.023 -> 0.015 (updated while chasing down attn bug)
* adam beta1 0.6 -> 0.8 (updated while chasing down attn bug)
* perform same batch size ramp of (131072, 262144, 393216), then continue and hold at 524288.
* perform same window size ramp of (3, 7, 11, 13), then continue up to 23
* split embed and head at 2/12 of training
* set backout lambda to 2/3 layers deep
* use the value embed and skip connection patterns from medium track
* Re-add second embedding from #124
* use all 16 attn layers to keep clean split on 8 GPU
* update gates to use 16 dims instead of 12 (same ratio to model_dim)
* Give attn gate and value embed gate 0.1 lr multiple (YouJiacheng mentioned this was effective on medium track a while back)
* I stop applying yarn once the long window extends beyond 13, as I don't have experience in how this behaves. This was updated while chasing down an attn bug and is likely not necessary.

Medium track features not yet integrated:
* Snoo and EMA optimizer wrappers
* Smear-MTP from #151
* Sharded mixed precision Muon from record 7.
* Cubic sliding window schedule (I just guessed a ramp that looked ok. flash attn 3 needs to compile each window size so I didn't want to incl every integer)
* The logit scaling formula from medium track- I wasn't sure how this was tuned so I just copied the short track formula.
* torch._inductor.config.coordinate_descent_tuning. I didn't notice this flag until I did the two runs. Presumably this is helpful.

I dropped the lr slightly when I was chasing down a bug where the attention window was larger than the max document length. The remaining parameters above are all first guesses based on my experience on the short track. Some of the features included are probably detrimental and don't scale below 3.28 loss, but on the whole the impact is strongly positive. I have no scaling intuition yet and the learning rate, weight decay, and schedules can be improved.

Overall I was happy to see that the learnings from the short track at 3.28 loss generally transfer to medium at 2.92 loss, giving a 25% runtime decrease with a subset of features and zero tuning. For reference, the full feature suite with maximum tuning gave a 33% runtime decrease on the short track.


## Timing and Validation
I only did 2 validation runs since this is well below the cutoff and can likely drop another 100 steps.
```
import scipy.stats
import torch

losses = [2.9165, 2.9166]
times = [1041.071, 1041.483]

print("p=%.4f" % scipy.stats.ttest_1samp(losses, 2.92, alternative="less").pvalue)
# p=0.0046

print("losses:", torch.std_mean(torch.tensor(losses)))
# losses: (tensor(7.0638e-05), tensor(2.9166))

print("time:", torch.std_mean(torch.tensor(times)))
# time: (tensor(0.2913), tensor(1041.2771))
```

timing prior record: 1379.574 
