This sets a new record for the speedrun through various optimizations to kernel speed. The improvement was found by my AI system, [Aster](https://www.asterlab.ai/)!

Here's a summary of the changes:
Kernel / Function | Change Type | Description
-- | -- | --
XXT_kernel | Memory | Switched to coalesced loads + in-register transpose (tl.trans).
ba_plus_cAA_kernel | Memory | Switched to coalesced loads + in-register transpose.
Host Configs (XXT, etc.) | Tuning | Increased num_warps from 4 to 8.
softcapped_entropy | Math | Replaced inner-loop divisions with pre-computed multiplications.
softcapped_entropy | Tuning | Increased BLOCK_SIZE from 1024 to 4096.

And some stats on the improvement:
<img width="500" height="250" alt="Screenshot 2026-02-02 at 4 23 52 AM" src="https://github.com/user-attachments/assets/a380af35-8e42-4a84-90a6-ca4a9263087a" />

```python
>>> from scipy import stats
>>> losses = [3.2763,3.2777,3.2794,3.2773,3.2774,3.279,3.279,3.2791,3.2767]
>>> print("p=%.4f" % stats.ttest_1samp(losses, 3.28, alternative="less").pvalue)
p=0.0004
```

Some additional things I think I should share:
- I ran all of these tests on a GPU but neglected to save the logs for them, and the GPU was provided through a cloud service where I do not have the ability to reclaim that specific GPU. I have since re-run on a different GPU and put the logs for that run in the logs folder.
- I was unable to run the existing code without pinning torch to 2.10.0 in the dockerfile (`RUN pip install --pre torch==2.10.0 --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade`) because of a fa3 prebuilt kernel not found issue


@KellerJordan , thank you so much for your work in organizing this speedrun competition!
