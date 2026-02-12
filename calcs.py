import scipy.stats
import torch

# First set (Retimed New Record group)
losses1 = [3.2783, 3.2777, 3.2783, 3.2792, 3.2786, 3.2778, 3.2803, 3.2788]
times1 = [91448, 91407, 91518, 91340, 91507, 91542, 91514, 91547]

# Second set (Retimed Current Record group)
losses2 = [3.2807, 3.2789, 3.2789, 3.2778, 3.2783, 3.2774, 3.2801, 3.2783]
times2 = [92394, 92359, 92203, 92367, 92421, 92252, 92389, 92252]

print("p=%.4f" % scipy.stats.ttest_1samp(losses1, 3.28, alternative="less").pvalue)
# p=0.0012
print("losses:", torch.std_mean(torch.tensor(losses1)))
# losses: (tensor(0.0008), tensor(3.2786))
print("time:", torch.std_mean(torch.tensor(times1, dtype=torch.float32)))
# time: (tensor(73.2694), tensor(91477.8750))

print("losses:", torch.std_mean(torch.tensor(losses2)))
# losses: (tensor(0.0011), tensor(3.2788))
print("time:", torch.std_mean(torch.tensor(times2, dtype=torch.float32)))
# time: (tensor(81.3843), tensor(92329.6250))