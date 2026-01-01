# Record 28th of August, 2025

Statistics about the measured results for the baseline and the updated run. [Full code here.](https://github.com/snimu/modded-nanogpt-experiments/tree/main/experiments/00003-value-embeddings)

## Baseline

Changelog:

- Removed the `torch._dynamo.config.compiled_autograd = True` flag (it caused flexattention errors)
- Removed the `_patched_trace_structured` function (it often causes errors)

### Final val_losses - baseline

Here's the list of final validation losses over 28 runs:

```python
[2.919321, 2.919348, 2.920103, 2.920455, 2.91936, 2.919165, 2.920336, 2.919816, 2.919747, 2.918579, 2.920533, 2.920729, 2.918076, 2.919458, 2.920759, 2.919738, 2.92073, 2.919327, 2.919639, 2.91942, 2.920585, 2.920464, 2.918828, 2.919279, 2.920514, 2.919351, 2.918162, 2.920809]
```

Here are some simple statistics:

- Mean:     2.920 ± 0.001
- Median:   2.920
- Min:      2.918
- Max:      2.921

Here are the t-test results:

```python
{
    "n": 28,
    "sample_mean": 2.9197368214285713,
    "sample_std": 0.0007835970267072948,
    "t_stat": -1.7772018695052398,
    "p_value": 0.04340171700038697,
    "alpha": 0.05,
    "decision": REJECT H0 (mean < threshold),
    "upper_conf_bound_mean": 2.9199890544627403,
    "threshold": 2.92
}
```

### Run times - baseline

Here are the raw run times over 28 runs:

```python
['1450.149', '1447.753', '1447.248', '1448.042', '1446.999', '1447.910', '1447.621', '1447.163', '1448.034', '1448.266', '1448.380', '1447.248', '1448.169', '1451.810', '1448.287', '1449.739', '1449.761', '1453.234', '1449.403', '1450.164', '1448.897', '1450.096', '1449.720', '1449.535', '1449.472', '1448.813', '1450.895', '1450.256']
```

And here are some simple statistics about the run times:

- Mean:     1449.038 ± 1.456
- Median:   1448.855
- Min:      1446.999
- Max:      1453.234

## Two new value embeddings

The actual record.

Changelog:

- Added two more value embeddings
  - Previously, the three value embeddings were applied to the layers: 0&13, 1&14, 2&15
  - Now, the five value embeddings are applied to the layers: 0&11, 1&12, 2&13, 3&14, 4&15
- Reduced step count from 5890 to 5820

### Final val_losses - record

The raw final validation losses over 37 runs:

```python
[2.919612, 2.919458, 2.918941, 2.917664, 2.91856, 2.919706, 2.919218, 2.918082, 2.919345, 2.920486, 2.919293, 2.917286, 2.921162, 2.919861, 2.917587, 2.919488, 2.919955, 2.919172, 2.919245, 2.918839, 2.918381, 2.919301, 2.917944, 2.919178, 2.918395, 2.920141, 2.918754, 2.918432, 2.919958, 2.91978, 2.919916, 2.919711, 2.918025, 2.919342, 2.920571, 2.917387, 2.919093]
```

Simple statistics:

- Mean:     2.919 ± 0.001
- Median:   2.919
- Min:      2.917
- Max:      2.921

T-test results:

```python
{
    "n": 37,
    "sample_mean": 2.919115378378378,
    "sample_std": 0.000915598388163916,
    "t_stat": -5.876968901489202,
    "p_value": 5.07368129288152e-07,
    "alpha": 0.05,
    "decision": REJECT H0 (mean < threshold),
    "upper_conf_bound_mean": 2.9193695067707086,
    "threshold": 2.92
}
```

### Run times - record

Raw run times:

```python
['1421.024', '1420.776', '1422.277', '1422.077', '1422.587', '1421.731', '1421.276', '1421.190', '1421.335', '1421.321', '1421.373', '1430.659', '1424.760', '1423.293', '1421.603', '1422.789', '1422.489',
'1455.587', '1421.598', '1424.514', '1425.991', '1423.341', '1444.257', '1465.063', '1428.880', '1430.782', '1435.003', '1426.705', '1423.921', '1424.339', '1423.867', '1423.950', '1424.241', '1467.321', 
'1424.330', '1424.331', '1424.449']
```

Simple statistics:

Mean:     1427.704 ± 11.387
Median:   1423.921
Min:      1420.776
Max:      1467.321
