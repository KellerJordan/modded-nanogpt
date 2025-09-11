This PR of 159.3s incorporates YaRN into the training window schedule and final validation. https://arxiv.org/pdf/2309.00071
This submission includes all recent WR improvements, including dropping initial MLP layer by @EmelyanenkoK in [PR 120](https://github.com/KellerJordan/modded-nanogpt/pull/120).

Longer attention windows take longer to train, but produce models with lower loss. Two phenomena occur in RoPE when the attention window is increased during or after training:
1. Dimensions with low frequency rotations experience unfamiliar rotation angles. For instance, a dimension that rotates 0.1 degrees per position will have experienced 0.1*384=38.4 degrees of rotation during training on ws 384. When the sliding window is expanded to 896, it experiences up to 89.6 degrees of rotation. This out of distribution data causes a temporary loss spike.
2. In particular when K and Q vectored are normed, perplexity of the attn mechanism increases as the number of keys increases. Applying a scaling factor d to softmax(d*QK) enables the perplexity of the data to be controlled as the number of keys in the attention window increases.

A single copy of rotary embeddings is stored in the model root to reduce update time, reduce memory size, and potentially improve cache performance.
```
# store single copy of rotary tensors
angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim//4, dtype=torch.float32)
# half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(head_dim//4)])
t = torch.arange(self.max_seq_len, dtype=torch.float32)
theta = torch.outer(t, angular_freq)
self.rotary_cos = nn.Buffer(theta.cos(), persistent=False)
self.rotary_sin = nn.Buffer(theta.sin(), persistent=False)
```

Based on empirical testing, the 0.1 constant in 0.1*log(curr/prev)+1 formula from YaRN is updated to 0.2.
The constant attn_scale of 0.12 is updated to a starting value of 0.1, such that the distribution over training has a similar mean, ranging between 0.1 and 0.14.
<img width="1333" height="388" alt="image" src="https://github.com/user-attachments/assets/1171ad00-084a-4e50-8d05-ef7c7730d7d6" />

```
# scale attention factor f in attn=softmax(f*qk) logarithmically with window size
windows = list(dict.fromkeys(args.ws_schedule + [args.ws_validate]))
scale_factors = [0.2 * math.log(curr / prev) + 1 for prev, curr in zip(windows[:-1], windows[1:])]
# start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
attn_scales = list(accumulate([0.1] + scale_factors, lambda acc, factor: acc * factor))
self.attn_scales = dict(zip(windows, attn_scales))
```

YaRN has a straighforward implementation, shown below. alpha and beta are left at the default constants of 1 and 32, based on the original YaRN paper which was tuned for Llama. The frequency update incurred by YaRN is most notable from ws 3->7 and dimensions 5 to 10.
<img width="1355" height="368" alt="image" src="https://github.com/user-attachments/assets/371b91a4-0ab7-4cbe-8c82-0f5f964f4022" />
```
def apply_yarn(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
    rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
    scaling_factor = old_window / new_window
    interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
    self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
    t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
    theta = torch.outer(t, self.angular_freq)
    self.rotary_cos.copy_(theta.cos())
    self.rotary_sin.copy_(theta.sin())
```



Arg ws_validate enables the model to be validated at a longer attention window than training. This arg is set to 13, which differs from the final training window size of 11.
<img width="951" height="517" alt="image" src="https://github.com/user-attachments/assets/f84dcc4b-b711-40d1-8309-120663994b80" />

```
def get_ws(step: int):
    if step == args.num_iterations:
        return args.ws_validate
    x = step / (1 + args.num_iterations)
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx]
```

Attention args are batched to improve readablility. cooldown_frac is increased from 0.45 to 0.5 to compliment the reduction from 1705 to 1670 steps, following the heuristic of a fixed number of cooldown steps. Dropping below 1695 steps has a secondary benefit of eliminating the 9th file read, saving roughly 200ms.

Without YaRN, there is a substantial spike in validation loss when the attention window is abrubtly increased from 3 to 7.
<img width="875" height="480" alt="image" src="https://github.com/user-attachments/assets/4c4a8dc5-3294-41a3-a152-57068314dd63" />

Extending the final validation window out shows roughly a 0.0015 improvement in loss for 11->13. Interestingly, odd increments perform substantially better. @varunneal has noted that "One thing to note is that floor division (ws_short = ws_long // 2) has different behavior for odd vs short window sizes. I generally found odd window sizes performed surprisingly better." The attention schedule follows (long/short) (3/1) -> (7/3) -> (11/5). It may be that the short attention window performs better when it is under 50% of the long window, or it may be that the model learns to fit the long/short ratio, and performs poorly when this ratio is substantially altered, or there may be a completely different explanation.

Ablations were ran to measure the impact of each change:
* new_record
* no_attn_scale. Keep constant attn scale of 0.12.
* no_freq_scale. Keep constant rotary freq based on 1024^(0..1).
* prior_record. Prior record with updated steps from 1705 to 1670 and cooldown frac to 0.5.
<img width="867" height="577" alt="image" src="https://github.com/user-attachments/assets/1ccd04f6-c118-4200-9d99-5b4697ba7061" />


Future Considerations:
* Right now model training is like a racecar with no brakes. There may be a way to effectively dampen the optimizer state momentum terms when the model updates its attention window size and 'changes direction'. Preliminary testing here on only the Muon params gave negative results.
* There may be a way to distribute the load of finding bos token indicies for all 8 files. If each GPU is given 1 file instead of 8 to locate the bos_tokens, this could save up to roughly 200ms*7 = 1.4 seconds assuming zero overhead.
* Starting RoPE at a max angular frequency of 1 radian per position, or 57 degrees, seems arbitrary. However, increasing this to 180 degrees did not show an improvement in performance.
* Plotting validation loss every 125 iterations masks critical issues like loss spikes on attn window updates. In general, more granular monitoring seems useful.

Validation:
```
import scipy.stats
import torch
accs = [3.2779, 3.2779, 3.2789, 3.2778, 3.2789, 3.2785, 3.2806]
times = [159.447, 158.998, 159.467, 159.191, 159.503, 159.259, 159.468]

print('p=%.4f' % scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue)
# p=0.0053

print('acc:',torch.std_mean(torch.tensor(accs)))
# acc: (tensor(0.0010), tensor(3.2786))

print('time:',torch.std_mean(torch.tensor(times)))
# time: (tensor(0.1897), tensor(159.3333))
```
