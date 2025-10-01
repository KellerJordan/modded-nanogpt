# 2025-10-01 Medium Track Record

I simply added the output of layer 11 (the 12'th layer) to the output of the final layer (layer 15, or the 16'th layer), in a weighted sum. The weights are scalars and are learned. This just means that right before applying the language head, I do this:

```python
skip_lambdas = self.scalars[-2:]
x = norm(x) * skip_lambdas[0] + norm(skip_connections[11]) * skip_lambdas[1]
```

Where the `skip_connections` contain the output latents of each layer, at the corresponding position.

These are the resulting final validation losses over 10 runs:

```python
[2.918849, 2.917008, 2.919394, 2.91776, 2.919522, 2.919678, 2.918775, 2.918111, 2.919395, 2.917938]
```

And here are simple stats about the final validation loss over these 10 runs:

- Mean: 2.9186
- Std: 0.0008985
- P-value: 0.0005035724941803675

The p-value is very low, so after only 10 runs we can be sure that the target loss is reached.

Now here are the corresponding run times:

```python
[1378.689, 1378.504, 1378.622, 1379.238, 1379.0, 1379.444, 1378.426, 1379.337, 1378.254, 1379.127]
```

And here are some simple stats about the times:

- Mean: 1378.8641 seconds ~= 22.98 minutes
- Std: 0.4177 seconds

The previous record's time is 1405.698 seconds, so this is a reduction of almost 27 seconds.
