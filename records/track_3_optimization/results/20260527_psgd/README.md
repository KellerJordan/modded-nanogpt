# Optimizer Baseline: PSGD

This is an implementation of PSGD.

## Details

The standard (whitening, kronecker) PSGD update can be summarized in psuedocode as:

```python
def update(W, g, Q0, Q1, m, step_t, lr, precond_lr, beta):
    """
      - Q0, Q1 are L/R preconditioner estimates (kronecker factors)
      - W, g, m are weight, gradient, momentum
    """

    # momentum w bias correction
    m.lerp_(g, 1 - beta)
    m_hat = m / (1 - beta ** step_t)

    # balance preconditioner factors
    scale_factor = (Q1.abs().amax() / Q0.abs().amax()).sqrt()
    Q0 *= scale_factor
    Q1 /= scale_factor

    # update preconditioners
    nu = randn_like(m_hat)
    v = randn_like(m_hat)
    #  A = Q0 U Q1^T, B = Q0^{-T} v Q1^{-1}
    A = Q0 @ (m_hat + torch.eps * m_hat.abs() * nu) @ Q1.T
    B = solve_triangular(Q1, solve_triangular(Q0.T, v, left=True), left=False)
    # gradient descent scaled by spectral norm
    Q0 -= (precond_lr / _spectral_norm(A@A.T + B@B.T)) * triu(A@A.T - B@B.T) @ Q0
    Q1 -= (precond_lr / _spectral_norm(A.T@A + B.T@B)) * triu(A.T@A - B.T@B) @ Q1

    # apply preconditioner
    u = (Q0.T @ Q0) @ m_hat @ (Q1.T @ Q1)

    # hyperball
    u = u * (W.norm() / u.norm())
    W_new = W - lr * u
    W = W_new * (W.norm() / W_new.norm())
```

I put learning rate on a linearly descending schedule for 100% of the run, starting at `lr=0.025`.
I also found optimal hyperparameters `precond_lr=1.0`, and `beta=0.95`.

Instead of trust-region clipping, which is common in many PSGD implementations, I've opted to just use Hyperball. Carefully tuned trust-region clipping + weight decay may outperform hyperball, but will introduce extra hyperparameters.

Many PSGD implementations use the previous iterate's spectral norm (or a rolling average) to smoothen or improve approximation quality, but I found that approximating it via power iterations each step is cheap and works well.


## Validation

At 3400 steps:

```python
import numpy as np

losses = [3.2764, 3.2777, 3.2774, 3.2750, 3.2768]
times = [611.1, 610.9, 610.8, 611.1, 611.7]

print(np.mean(losses))
# 3.2766600000000006
print((3.28 - 3.09 * 0.0013 / np.sqrt(5)))
# 3.2782035429868763
```

The step count can likely be safely decreased by 20+ steps.


## References

The following repositories served as important references:
- [HomebrewML/HeavyBall](https://github.com/HomebrewML/HeavyBall)
- [evanatyourservice/distributed_kron](https://github.com/evanatyourservice/distributed_kron)
- [evanatyourservice/psgd_jax](https://github.com/evanatyourservice/psgd_jax)
- [evanatyourservice/kron_torch](https://github.com/evanatyourservice/kron_torch)
- [lixilinx/psgd_torch](https://github.com/lixilinx/psgd_torch)
- [PSGD_Nuon](https://github.com/opooladz/PSGD_Nuon)


As well as the original PSGD papers, including
1. [Xi-Lin Li (2015)](https://arxiv.org/abs/1512.04202) — Preconditioned Stochastic Gradient Descent
2. [Xi-Lin Li (2024)](https://arxiv.org/abs/2402.11858) — Stochastic Hessian Fittings with Lie Groups
3. [Pooladzandi & Li (2024)](https://arxiv.org/abs/2402.04553) — Curvature-Informed SGD via General Purpose Lie-Group Preconditioners

This submission would be totally impossible without the work of Xi-Lin Li, Omead Pooladzandi, Evan Walters, and Lucas Nestler.
