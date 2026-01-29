# Tie First and Last VEs

Swapping the first two VEs reduces validation loss.

Our long-standing pattern of value embeddings has been:

`12------012`

Switching this to:

`21------012`

reduced loss by -0.0012

Consistent with the previous PR, I found that untying VE1 reduces it further as well.

If we rename VE2 to VE0, it becomes:

`01------230`

```
                             Runs  Time μ  Time σ  Time +/-  Time p  Loss μ  Loss σ  Loss +/-      p
baseline                        6 98.3442  0.0686    0.0000     NaN  3.2785  0.0013    0.0000 0.0184

ve_21------012 w/o PolarT       4 98.3590  0.0868    0.0148  0.6077  3.2774  0.0010   -0.0011 0.0077
ve_21------012                  4 98.3490  0.0336    0.0048  0.5569  3.2772  0.0011   -0.0012 0.0079
ve_21------012 -5stps           6 98.1192  0.0400   -0.2250  0.0001  3.2786  0.0004    0.0001 0.0002 - Remove more steps?

ve_01------230 -15stps          4 98.0825  0.0896   -0.2617  0.0018  3.2769  0.0010   -0.0016 0.0040 - Best so far

ve_210-----012                  4 98.5325  0.0455    0.1883  0.9996  3.2764  0.0010   -0.0021 0.0024
ve_210-----012 -10stps          6 97.8797  0.0898   -0.4645  0.0000  3.2791  0.0014    0.0006 0.0899 (Fail)
ve_210-----012 untie, -5stps    4 98.4843  0.1030    0.1401  0.9677  3.2792  0.0016    0.0007 0.1820 (Fail)
```

**Rationale**

Trying to make sense of why this unique VE pattern works well lead me to speculate that VE2 (which we've had on our second and last layer, tied) might be more semantic, and therefore relate well to both the first and last layers.

This also might offer some explanation for why shifting VE2 deeper into the model (e.g., via `012-----012`) hasn't worked.

## Polar Express Transpose

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

I also added a fallback for the fused ReLU kernel, but need to review that change further.
