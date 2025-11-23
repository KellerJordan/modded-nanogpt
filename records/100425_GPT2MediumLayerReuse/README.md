I simply added the output of layer 11 (the 12'th layer) to the output of the final layer (layer 15, or the 16'th layer), in a weighted sum. The weights are scalars and are learned. This just means that right before applying the language head, I do this:

```python
skip_lambdas = self.scalars[-2:]
x = norm(x) * skip_lambdas[0] + norm(skip_connections[11]) * skip_lambdas[1]
```

Where the `skip_connections` contain the output latents of each layer, at the corresponding position.

Doing this allowed me to reduce the step count from 5590 to 5550, which lead to the following results.

The baseline comes from PR#137. I re-ran the code for 5 runs and got the following results:

- Mean final validation loss: 2.9191
- Mean time: 1393.16 ~= 23.22 minutes
- T-test p-value: 0.01194797508928048

These are the resulting final validation losses over 10 runs:

```python
[2.919485, 2.918384, 2.918878, 2.918476, 2.920099, 2.919609, 2.918705, 2.91872, 2.919772, 2.918594, 2.917798, 2.919295, 2.920676, 2.919743, 2.920052, 2.919843, 2.920081, 2.919675, 2.919486, 2.919177, 2.919529, 2.919678]
```

And here are simple stats about the final validation loss over these 10 runs:

- Mean: 2.9194
- Std: 0.00069
- P-value: 0.0001256

Now here are the corresponding run times:

```python
[1384.256, 1384.324, 1384.185, 1383.412, 1392.184, 1392.305, 1383.552, 1383.785, 1383.811, 1383.785, 1383.434, 1383.753, 1383.082, 1383.284, 1383.827, 1385.682, 1383.579, 1383.422, 1383.467, 1385.108, 1383.398, 1384.058]
```

And here are some simple stats about the times:

- Mean: 1384.6224 seconds ~= 23.08 minutes
- Std: 2.5382 seconds

This is a reduction in final run time of ~8.54 seconds.
