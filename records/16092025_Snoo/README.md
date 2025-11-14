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

```
print(df_nanogpt_med_5640[['loss', 'train_time']].reset_index(drop=True))
print(f"{df_nanogpt_med_5640['train_time'].mean()=}")
print(f"{df_nanogpt_med_5640['train_time'].std()=}")
print(f"{df_nanogpt_med_5640['train_time'].count()=}")
print(f"{df_nanogpt_med_5640['train_time'].min()=}")
print(f"{df_nanogpt_med_5640['train_time'].max()=}")
print(f"{df_nanogpt_med_5640['loss'].mean()=}")
print(f"{df_nanogpt_med_5640['loss'].min()=}")
print(f"{df_nanogpt_med_5640['loss'].max()=}")
print(f"{df_nanogpt_med_5640['loss'].std()=}")
print(f"{scipy.stats.ttest_1samp(df_nanogpt_med_5640['loss'].to_numpy().tolist(), 2.92, alternative='less').pvalue=}")
```

| Train time | Val Loss |
| ---------  | -------- |
| 2.920866   | 1404429  |
| 2.919848   | 1405527  |
| 2.920058   | 1405527  |
| 2.919486   | 1404794  |
| 2.919221   | 1404749  |
| 2.919295   | 1403139  |
| 2.920905   | 1402907  |
| 2.919472   | 1403525  |
| 2.919485   | 1410229  |
| 2.918337   | 1403318  |
| 2.918905   | 1403348  |
| 2.921223   | 1403409  |
| 2.919060   | 1403128  |
| 2.919060   | 1405181  |
| 2.919242   | 1405181  |
| 2.918973   | 1403238  |
| 2.919746   | 1403648  |
| 2.919541   | 1403085  |
| 2.919301   | 1403406  |
| 2.919458   | 1402836  |
| 2.919502   | 1402963  |
| 2.917969   | 1403876  |
| 2.920135   | 1402608  |
| 2.920262   | 1404008  |
| 2.919758   | 1403351  |
| 2.920201   | 1403465  |
| 2.919559   | 1403964  |
| 2.920641   | 1402720  |
| 2.919203   | 1406409  |
| 2.919769   | 1402930  |

df_nanogpt_med_5640['train_time'].mean()=np.float64(1404029.9333333333)
df_nanogpt_med_5640['train_time'].std()=1523.797296985019
df_nanogpt_med_5640['train_time'].count()=np.int64(30)
df_nanogpt_med_5640['train_time'].min()=1402608.0
df_nanogpt_med_5640['train_time'].max()=1410229.0
df_nanogpt_med_5640['loss'].mean()=np.float64(2.919616033333334)
df_nanogpt_med_5640['loss'].min()=2.917969
df_nanogpt_med_5640['loss'].max()=2.921223
df_nanogpt_med_5640['loss'].std()=0.0007156313095892154
scipy.stats.ttest_1samp(df_nanogpt_med_5640['loss'].to_numpy().tolist(), 2.92, alternative='less').pvalue=np.float64(0.0032015979421488247)
