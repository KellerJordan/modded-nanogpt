# Modded-NanoGPT

This is a variant of the [PyTorch GPT-2 trainer](https://github.com/karpathy/llm.c/blob/7b929300217ff1a974b63791a228928b39b26409/train_gpt2.py) from
Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) repo. It:
* Trains 3x more efficiently (taking only 3.25B tokens instead of 10B to reach the same validation loss).
* Has shorter code (524 lines instead of 860).
* Implements architectural modernizations (rotary embeddings and RMSNorm).
* Implements a new optimizer (Muon - Momentum Orthogonalized by Newton-schulz).

To execute the training, run the following three commands on an 8xA100 or 8xH100 node.
They complete in <45min on an 8xH100 with decent internet connection.
```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py
./run.sh
```

This will train a 124M-parameter transformer for 6200 steps on 3.25B tokens of Fineweb [1], achieving ~3.278 validation loss.
For comparison, the default llm.c PyTorch trainer yields [>3.28 validation loss after training for 10B tokens](https://github.com/karpathy/llm.c/discussions/481).

---

## Figures

Figure 1. Proposed optimizer vs. a well-tuned AdamW.
![](img/fig_optimizer.png)

---

## Proposed optimizer

For this training scenario, the proposed optimizer has the following properties:
* Half the memory usage of Adam
* 1.5x faster training
* <9% wallclock overhead (which can be further brought down by distributing the overhead; it's currently performed redundantly on all 8 GPUs)

The optimizer is defined as follows:

![](img/algo_optimizer.png)

Where NewtonSchulz5 is the following Newton-Schulz iteration [2, 3]:
```python
@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T 
    for _ in range(steps):
        A = X @ X.T 
        B = A @ X 
        X = a * X + b * B + c * A @ B 
    if G.size(0) > G.size(1):
        X = X.T 
    return X.to(G.dtype)
```

### Provenance

Many of the choices made to generate this optimizer were obtained experimentally by our pursuit of [CIFAR-10 speedrunning](https://github.com/KellerJordan/cifar10-airbench).
In particular, we experimentally obtained the following practices:
* Using Nesterov momentum inside the update, with orthogonalization applied after momentum.
* Using a specifically quintic Newton-Schulz iteration as the method of orthogonalization.
* Using non-convergent coefficients for the quintic polynomial in order to maximize slope at zero, and thereby minimize the number of necessary Newton-Schulz iterations.
* Running the Newton-Schulz iteration in bfloat16 (whereas Shampoo implementations often compute the preconditioners via inverse-pth-roots in fp32 or fp64).

Our use of a Newton-Schulz iteration for orthogonalization traces to [Bernstein & Newhouse (2024)](https://arxiv.org/abs/2409.20325),
who suggested it as a way to compute Shampoo [5, 6] preconditioners, and theoretically explored Shampoo without preconditioner accumulation.
In particular, Jeremy Bernstein @jxbz sent us the draft, which caused us to experiment with various Newton-Schulz iterations as the
orthogonalization method for this optimizer.
If we had used SVD instead of a Newton-Schulz iteration, this optimizer would have been too slow to be useful.
Bernstein & Newhouse also pointed out that Shampoo without preconditioner accumulation is equivalent to steepest descent in the spectral norm,
and therefore Shampoo can be thought of as a way to smooth out spectral steepest descent.
The proposed optimizer can be thought of as a second way of smoothing spectral steepest descent, with a different set of memory and runtime tradeoffs
compared to Shampoo.

---

## Other general differences between this codebase and NanoGPT

To simplify the code, some features have been removed, including text generation.
And to obtain a training speed improvement, we have diverged from being a strict reproduction of the GPT-2 paper.

The speedup is due to the following changes:
- Increased learning rate by 3x
- Switched to trapezoidal learning rate schedule following [7]
- Switched to rotary embeddings
- Removed the special initialization for linear layers before residuals. Instead, just scale down the output of the attention block by a fixed scalar.
- Removed all affine scale and bias parameters from the architecture, and switched to RMSNorm (actually this causes a slight slowdown, and I just did it to reduce code complexity)
- Switched from AdamW to new optimizer, and removed learning rate warmup

---

## References

1. [Penedo, Guilherme, et al. "The fineweb datasets: Decanting the web for the finest text data at scale." arXiv preprint arXiv:2406.17557 (2024).](https://arxiv.org/abs/2406.17557)
2. Nicholas J. Higham. Functions of Matrices. Society for Industrial and Applied Mathematics, 2008. Equation 5.22.
3. Günther Schulz. Iterative Berechnung der reziproken Matrix. Z. Angew. Math. Mech., 13:57–59, 1933.
4. [Jeremy Bernstein and Laker Newhouse. "Old Optimizer, New Norm: An Anthology." arxiv preprint arXiv:2409.20325 (2024).](https://arxiv.org/abs/2409.20325)
5. [Vineet Gupta, Tomer Koren, and Yoram Singer. "Shampoo: Preconditioned stochastic tensor optimization." International Conference on Machine Learning. PMLR, 2018.](https://arxiv.org/abs/1802.09568)
6. [Anil, Rohan, et al. "Scalable second order optimization for deep learning." arXiv preprint arXiv:2002.09018 (2020).](https://arxiv.org/abs/2002.09018)
7. [Hägele, Alexander, et al. "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations." arXiv preprint arXiv:2405.18392 (2024).](https://arxiv.org/abs/2405.18392)

<img src="img/dofa.jpg" alt="itsover_wereback" style="width:100%;">

```
@software{moddednanogpt2024,
    author={Jordan, Keller},
    title={Modded-NanoGPT},
    url={https://github.com/KellerJordan/Modded-NanoGPT},
    version={0.1.0},
    year = {2024}
}
```

