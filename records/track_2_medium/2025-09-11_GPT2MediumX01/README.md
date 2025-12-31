# Add additional input embedding

Includes [PR#119](https://github.com/KellerJordan/modded-nanogpt/pull/119).

Previously, modded-nanogpt medium added x0 to the residual at the input of every layer:

```python
# GPT
def forward(self, input_seq, ...):
    ...

    x = x0 = norm(self.embed(input_seq[None]))

    ...

    for i in range(len(self.blocks)):
        ...
        x = self.blocks[i](x, x0, lambdas, ...)
        ...

# Block
def forward(self, x, x0, lambdas, ...):
    x = lambdas[0] * x + lambdas[1] * x0
```

Where `lambdas` are learned scalars.

This update adds another embedding module and adds it at every layer:

```python
# GPT
def forward(self, input_seq, ...):
    ...

    x = x00 = norm(self.embed1(input_seq[None]))
    x01 = norm(self.embed2(input_seq[None]))

    ...

    for i in range(len(self.blocks)):
        ...
        x = self.blocks[i](x, x00, x01, lambdas, ...)
        ...

# Block
def forward(self, x, x00, x01, lambdas, ...):
    x = lambdas[0] * x + lambdas[1] * x00 + lambdas[2] * x02
```

While this slows down training, it increases learning per step, thus allowing us to reduce the step count to 5690.

Here are the resulting final validation losses over 19 runs:

```python
[2.919502, 2.91976, 2.920582, 2.919331, 2.919008, 2.919827, 2.918785, 2.918519, 2.919297, 2.920061, 2.918938, 2.919342, 2.918186, 2.920546, 2.91954, 2.919093, 2.918951, 2.919599, 2.919956]
```

And these are the basic stats:

- Mean: 2.9194117368421053
- Median: 2.919342
- Std: 0.000613243648848653
- Min: 2.918186
- Max: 2.920582

And t-test results:

```python
{
    'n': 19,
    'sample_mean': 2.9194117368421053,
    'sample_std': 0.0006300479560324045,
    't_stat': -4.069816643193549,
    'p_value': 0.00035946114919240566,
    'alpha': 0.05,
    'decision': 'REJECT H0 (mean < threshold)',
    'upper_conf_bound_mean': 2.919662383449226,
    'threshold': 2.92
}
```

The final loss is below 2.92 with >99% likelihood.

Here are the corresponding run-times in seconds:

```python
[1414.299, 1412.033, 1411.668, 1421.735, 1411.998, 1411.094, 1412.637, 1410.047, 1410.509, 1412.048, 1411.574, 1415.299, 1411.649, 1412.94, 1412.508, 1410.912, 1415.296, 1410.778, 1407.511]
```

Leading to the following stats:

- Mean: 1412.4492105263155
- Median: 1411.998
- Std: 2.8062021268488864
- Min: 1407.511
- Max: 1421.735

The mean time is ~1412.5 seconds, or 23.54 minutes.
