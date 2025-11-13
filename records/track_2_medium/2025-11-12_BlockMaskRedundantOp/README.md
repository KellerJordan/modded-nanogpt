# Remove a redundant tensor op in block mask calculation

The change is super simple. The function `create_blockmasks` performs a series of tensor operations to ultimately create a block mask containing the indices of "partial" blocks and "full blocks" which indicate to flex attention where to apply the `document_causal` mask_mod and where to skip it respectively.

One of those tensor ops is redundant, saving about 220 ms overall.

## Timing
We can see in the record that it completes in 1496063 ms. When I timed it against the unmodified code, it finished in 1496283 ms so this change should save about 220 ms.


## The redundant operation
To create block masks, we first aim to create two masks - `blockmask_any` and `blockmask_all` and later find the partial blocks by computing `blockmask_any & ~blockmask_all`.

Focusing on `blockmask_any`, it's computed like so:
```
document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
blockmask_any = causal_blockmask_any & document_blockmask_any
```

The idea is that this is exactly equal to
```
blockmask_any = causal_blockmask_any & (docs_low[:, None] <= docs_high)
```
and thus the `&` operation to compute `document_blockmask_any` is redundant.

## Why is the op redundant?
This is a little confusing but here is my attempt:

Consider the `(i, j)` element in the matrix `document_blockmask_any`. Since we are going to apply causal mask later to compute `blockmask_any`, all elements where `j > i` will become `False`. So we only care about `j <= i`.

If `j <= i`, the ending doc of block i must be greater than the starting doc of block j by definition since block j appears before block i. So `docs_high[:, None] >= docs_low` must evaluate to `True` for `j <= i`.

Another way to view this is, `docs_high[i] >= docs_low[j]` for all `j <= i` therefore, for `j <= i`, computing `docs_high[:, None] >= docs_low` is redundant.

Note that, on the other hand, if `j <= i`, then `docs_low[i]` is not guaranteed to be smaller than `docs_high[j]`.
This can be seen via a counter-example, let's say the `docs` vector is like so:
```
1 1 | 1 2 | 2 2 | 2 2 | 3 3 | 3 4 ...
```
Consider a block size of 2 which yields `docs_low` and `docs_high` as:
```
# docs_low  (starting doc for each block)
1 1 2 2 3 3

# docs_high  (ending doc for each block)
1 2 2 2 3 4
```
Then for `i = 4, j = 1` (i.e. `j <= i`), `docs_low[i] = 3` while `docs_high[j] = 2`.
We can also verify in this example that for all `j <= i`, `docs_high[i] >= docs_low[j]`.