## Summary
This PR builds on PR#124 and adds the Snoo optimizer (Sparse Nesterov Outer Optimizer) which improves the medium WR by 60 steps, (~10s).
Snoo is a look-ahead momentum-based wrapper that can improve the quality of large language models (LLM) and other models. Snoo implicitly smoothens the training trajectory and instills a bias towards flatter minima. Snoo is computationally efficient, incurring minimal overhead in compute and moderate memory usage.
## Code
```python
@torch.no_grad()
def step(
    self,
) -> None:
    if self.current_step % self.k == 0:
        for p_new, p_old in zip(self.model_params, self.outer_buf):
            p_new.grad = p_old.data - p_new.data
            p_new.copy_(p_old, non_blocking=True)
        self.optimizer.step()
        for p_new, p_old in zip(self.model_params, self.outer_buf):
            p_old.copy_(p_new, non_blocking=True)
    self.current_step += 1
```

## Stats:
| Train time| Val Loss     |
| --------- | ------------ |
|	1425323	| 2.9200130000 |
|	1399753	| 2.9192430000 |
|	1400871	| 2.9196530000 |
|	1432667	| 2.9202780000 |
|	1400392	| 2.9209690000 |
|	1399248	| 2.9187820000 |
|	1399035	| 2.9204980000 |
|	1399037	| 2.9198950000 |
|	1399258	| 2.9191080000 |
|	1399369	| 2.9205680000 |
|	1398911	| 2.9203090000 |
|	1399043	| 2.9192930000 |
|	1398410	| 2.9185420000 |
|	1398703	| 2.9196380000 |
|	1399135	| 2.9197590000 |
|	1399177	| 2.9176600000 |
|	1398529	| 2.9196670000 |
|	1398326	| 2.9213670000 |
|	1398652	| 2.9195370000 |
|	1399108	| 2.9179940000 |
|	1398671	| 2.9200720000 |
|	1398786	| 2.9207020000 |
|	1423478	| 2.9210300000 |
|	1398958	| 2.9192110000 |
|	1398334	| 2.9189310000 |
|	1431761	| 2.9182240000 |
|	1398236	| 2.9214560000 |
|	1398511	| 2.9195970000 |
|	1397775	| 2.9194410000 |
|	1398159	| 2.9199210000 |
|	1398501	| 2.9211470000 |
|	1398141	| 2.9178630000 |
|	1397903	| 2.9186550000 |
|	1399010	| 2.9195980000 |
|	1397790	| 2.9190140000 |
|	1398512	| 2.9209280000 |
|	1398639	| 2.9202180000 |
|	1398457	| 2.9196910000 |
|	1398093	| 2.9195350000 |
|	1397701	| 2.9185550000 |
|	1397840	| 2.9196240000 |
|	1397735	| 2.9193710000 |

Count: 42
Train Time:
- Mean: 1401522.33
- Std: 8906.82
- Min: 1397701
- Max: 1432667

Val Loss:
- Mean: 2.919656
- Std: 0.0009364946890159557
- Min: 2.91766
- Max: 2.921456
- P_Val(<2.92): 0.011
