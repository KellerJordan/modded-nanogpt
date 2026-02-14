# Transpose Kernels

I've consolidated all of the recent PRs in this submission, and also added a few kernel improvements:
1. Transpose copy and add kernels to speed up the lm head and embedding interaction.
2. Eliminated a backwards select kernel
3. Transposed polar express from my previous abandoned PR

## Transpose Copy

Transposing the LM head (in [PR200](https://github.com/KellerJordan/modded-nanogpt/pull/200)) eliminated an expensive gradient accumulation kernel that pytorch would run every step, but the benefit was partially negated because we still had to do some element-wise operations to artificially keep the embeddings and LM head tied. 

My rationale at the time was that this was fine because it would be overlapped with compute, but the trace files have had some glaring communication gaps caused by these steps. I learned it's because the kernels were using all of the SMs, preventing the GPUs from being able to work on communication at the same time. (The fact that communication speed relates to SM availability seems like an interesting insight).

Claude was able to write efficient "transpose add" and "transpose copy" kernels which reduced the cost of these steps significantly. They still fully utilize the GPU, preventing overlap, but it's more minor now.

## "Select Backwards" Kernels

I noticed recently that my/our efforts to avoid the select-backwards kernels caused by the parameter banks haven't been entirely working--the trace file is still full of them. 

Claude was able to eliminate these by changing the way in which the weights are accessed. 

Curiously, one of the three fixes had a clear, consistent negative impact despite being mathematically equivalent. It must be due to a difference in the compiler's choice of kernel, and the order of operations in it. I had to leave it out. 

Another of the three seemed like it might be hurting loss--less conclusive--but I left it out.

We were able to fix what seemed to be the biggest offender though, surrounding the MLP weights, without impacting loss.

## Polar Express Transpose

(Copied from my closed PR)

I've also included an improvement to the Polar Express kernel. It was written to work on "wide" matrices, but we've had our MLP weights stored vertically. The current Polar Express code handles this by transposing the matrix, resulting in a slow element-wise kernel for the Nesterov momentum step. 

The algebra can be re-ordered instead to allow for working directly on "tall" matrices:

```
# Polar express
X = g.bfloat16()

X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

if g.size(-2) > g.size(-1): # Tall matrix
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B

else: # Wide matrix (original math)
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

return X
```

I had Claude write an XTX variant and verified that it was numerically equivalent.

The speed improvement is likely too small to be measurable, but it's clear in the trace files.

I came across it while exploring nanochat--here's the impact this fix has on a much larger (25-layer) model:

## A100 Support

I added a few things to make our code easier for anyone on a < H100 GPU.

It checks the compute capability to automate the decision on FP8, and also falls back to an FA2 implementation using the same interface we have for FA3 (there's a repo similar to Varun's for retrieving pre-built FA2).

## Worse Loss without ML Changes

I keep running into this issue of mathematically equivalent modifications having a negative impact on loss. The one place where I could see that being a legitimate bug is if I inadvertently change the precision of a parameter or an operation. 

It seems that the placement of kernel boundaries and the order of operations in those kernels has the potential to create problems. 

This has foiled a number of my recent attempts to make improvements. The select backwards kernels are a good example. I also tried in my last PR to break out all of the scalars, and pack them in the optimizer instead of having them packed in the model. But it hurt loss!

The sparse bigram table change seems like it may have the same problem--I saw an increase in loss after integrating it. I didn't dig into it to confirm that there isn't a legitimate problem, but since the impact was small my guess is it's the same kind of issue I've been running into.

The VE tuning changes balanced it (but without removing any steps). 

## Consolidating PRs

I noticed that none of the recent PRs build on one another. I went through and integrated them for this PR. The learning schedule changes from "station" needed a small change to make them adapt to the current step count.


# ===============


# Transpose Kernels

I've consolidated all of the recent PRs in this submission, and also added a few kernel improvements:
1. Transpose copy and add kernels to speed up the lm head and embedding interaction.
2. Eliminated a backwards select kernel
3. Transposed polar express from my previous abandoned PR

```

                                   Runs   Time μ  Time σ  Time +/-  Time p  Loss μ  Loss σ  Loss +/-      p
Consolidated Prior PRs                4  90.0190  0.0236    0.0000     NaN  3.2790  0.0013    0.0000 0.1120
This PR                               4  89.6460  0.0857   -0.3730  0.0010  3.2776  0.0010   -0.0014 0.0082

everything:
  losses = [3.2764, 3.2774, 3.2780, 3.2787]
  times  = [89.7110, 89.5220, 89.6940, 89.6570]
  (n = 4)

baseline:
  losses = [3.2808, 3.2782, 3.2791, 3.2780]
  times  = [90.0200, 90.0410, 89.9860, 90.0290]
  (n = 4)
```


## Transpose Copy

Transposing the LM head (in [PR200](https://github.com/KellerJordan/modded-nanogpt/pull/200)) eliminated an expensive gradient accumulation kernel that pytorch would run every step, but the benefit was partially negated because we still had to do some element-wise operations to artificially keep the embeddings and LM head tied. 

My rationale at the time was that this was fine because it would be overlapped with compute, but the trace files have had some glaring communication gaps caused by these steps. I learned it's because the kernels were using all of the SMs, preventing the GPUs from being able to work on communication at the same time. (The fact that communication speed relates to SM availability seems like an interesting insight).

Claude was able to write efficient "transpose add" and "transpose copy" kernels which reduced the cost of these steps significantly. They still fully utilize the GPU, preventing overlap, but it's more minor now.

## "Select Backwards" Kernels

I noticed recently that my/our efforts to avoid the select-backwards kernels caused by the parameter banks haven't been entirely working--the trace file is still full of them. 

Claude was able to eliminate these by changing the way in which the weights are accessed. 

Curiously, one of the three fixes had a clear, consistent negative impact despite being mathematically equivalent. It must be due to a difference in the compiler's choice of kernel, and the order of operations in it. I had to leave it out. 

Another of the three seemed like it might be hurting loss--less conclusive--but I left it out.

We were able to fix what seemed to be the biggest offender though, surrounding the MLP weights, without impacting loss.

## Polar Express Transpose

(Copied from my closed PR)

I've also included an improvement to the Polar Express kernel. It was written to work on "wide" matrices, but we've had our MLP weights stored vertically. The current Polar Express code handles this by transposing the matrix, resulting in a slow element-wise kernel for the Nesterov momentum step. 

The algebra can be re-ordered instead to allow for working directly on "tall" matrices:

```
# Polar express
X = g.bfloat16()

X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

if g.size(-2) > g.size(-1): # Tall matrix
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B

else: # Wide matrix (original math)
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

return X
```

I had Claude write an XTX variant and verified that it was numerically equivalent.

The speed improvement is likely too small to be measurable, but it's clear in the trace files.

I came across it while exploring nanochat--here's the impact this fix has on a much larger (25-layer) model:

## Worse Loss without ML Changes

I keep running into this issue of mathematically equivalent modifications having a negative impact on loss. The one place where I could see that being a legitimate bug is if I inadvertently change the precision of a parameter or an operation. 

It seems that the placement of kernel boundaries and the order of operations in those kernels has the potential to create problems. 

This has foiled a number of my recent attempts to make improvements. The select backwards kernels are a good example. I also tried in my last PR to break out all of the scalars, and pack them in the optimizer instead of having them packed in the model. But it hurt loss!

The sparse bigram table change seems like it may have the same problem--I saw an increase in loss after integrating it. I didn't dig into it to confirm that there isn't a legitimate problem, but since the impact was small my guess is it's the same kind of issue I've been running into.

The VE tuning changes balanced it (but without removing any steps). 

## Consolidating PRs

I noticed that none of the recent PRs build on one another. I went through and integrated them for this PR. The learning schedule changes from "station" needed a small change to make them adapt to the current step count.
