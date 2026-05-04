# Contra Muon + SOAP MLP pre-conditioning results

Before this change, all hidden 2D matrices (attention Q/K/V/O and MLP
fc/proj) followed the same path: Nesterov momentum → Newton-Schulz
orthogonalization → contra subtraction → NorMuon-lite row variance
normalization → u/w floor.

Attention matrices still follow that path. For MLP weights (`mlp.fc.weight`
and `mlp.proj.weight`) the pipeline is now: Nesterov momentum →
**Shampoo-basis preconditioning** → Newton-Schulz orthogonalization →
contra subtraction → NorMuon-lite row variance normalization → u/w floor.

Concretely, for an MLP matrix gradient `G_t`, first form the usual
Muon/Nesterov momentum:

```math
M_t = \mu M_{t-1} + (1-\mu)G_t,\qquad U_t = (1-\mu)G_t + \mu M_t.
```

Then maintain Shampoo-style row/column covariances
`L_t = EMA(G_t G_t^T)` and `R_t = EMA(G_t^T G_t)`, and compute their
eigenvector matrices `Q_r = eig(L_t)`, `Q_c = eig(R_t)`. These define
a rotated coordinate system aligned with the principal gradient directions.
Run Adam-style second-moment scaling in that basis:

```math
\hat U_t = Q_r^T U_t Q_c,\qquad
V_t = \beta_2 V_{t-1} + (1-\beta_2)\hat U_t^2,
```

```math
\tilde U_t = Q_r\left(\frac{\hat U_t}{\sqrt{V_t}+\epsilon}\right)Q_c^T,
\qquad
\tilde U_t \leftarrow \tilde U_t\frac{\|U_t\|_F}{\|\tilde U_t\|_F}.
```

The preconditioned momentum `\tilde U_t` is then passed through the usual
Contra-NorMuon update.

Everything else follows PR 275 Contra Muon with NorMuon-lite with row/column variance normalization and u/w-floor postprocessing.

Note: the archived scripts/logs print `muon_weight_decay=0.025`, but `Muon.step`
does not use the stored `weight_decay` param-group value. The effective Muon
weight decay for these runs is therefore zero.

Across 4 non-cherry-picked runs, the step 3150 mean validation loss is 3.27758750. Under the Track 3 README's one-sided z-test with `sigma=0.0013`, this gives `z=3.7115` and `p=1.03e-4`. No tuning was done beyond the SOAP betas, so some tuning + more runs could probably make this -100 steps. I tried w/ all params initially, but it didn't work (probably needs retuning).

| Run | 3125 val | 3150 val | 3175 val |
| -: | -: | -: | -: |
| 1 | 3.27861 | 3.27720 | 3.27659 |
| 2 | 3.27907 | 3.27764 | 3.27703 |
| 3 | 3.27897 | 3.27754 | 3.27691 |
| 4 | 3.27944 | 3.27797 | 3.27735 |
| **Mean** | **3.27902250** | **3.27758750** | **3.27697000** |
