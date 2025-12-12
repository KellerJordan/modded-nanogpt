import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.distributed as dist

# Optional torch.compile compatibility
if hasattr(torch, "compile"):
    _compile = torch.compile
else:  # pragma: no cover
    def _compile(f):
        return f


def _as_full_prec_tensor(val: float, device: torch.device) -> Tensor:
    """
    Helper that mimics torch._as_tensor_fullprec where available, but safely
    falls back to a standard float32 tensor otherwise.
    """
    if hasattr(torch, "_as_tensor_fullprec"):
        # type: ignore[attr-defined]
        return torch._as_tensor_fullprec(val)  # pragma: no cover
    return torch.tensor(val, dtype=torch.float32, device=device)


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """
    Reference Muon Newton–Schulz quintic iteration (unchanged).
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Quintic NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


@_compile
def _muon_update_kernel(
    acc_bf16_view_u16: Tensor,
    mantissa: Tensor,
    momentum_buffer: Tensor,
    grad: Tensor,
    momentum: Tensor,
    eff_lr: Tensor,
    eff_weight_decay: Tensor,
) -> None:
    """
    Bit-for-bit equivalent to the reference Muon update (kept intact).
    """
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    # Same two-step use of momentum as in the reference implementation
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)

    acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    acc_view_f32 = acc_m_u32.view(torch.float32)
    acc_view_f32.mul_(1 - eff_weight_decay)
    acc_view_f32.add_(other=v, alpha=-eff_lr)
    acc_m_u32 = acc_view_f32.view(torch.uint32)
    acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    mantissa.copy_(acc_m_u32.to(torch.uint16))


# ---- Geometry helpers (used only when spectral gating is enabled) ----

def power_iteration_smax(A: Tensor, iters: int = 2) -> Tensor:
    """
    Approximate largest singular value of a (batched) 2D tensor via power iteration.
    """
    A = A.float()
    if A.ndim == 2:
        m, n = A.shape
        v = torch.randn(n, device=A.device, dtype=A.dtype)
        v = v / (v.norm() + 1e-8)
        for _ in range(iters):
            u = A @ v
            u = u / (u.norm() + 1e-8)
            v = A.mT @ u
            v = v / (v.norm() + 1e-8)
        sigma = (A @ v).norm()
        return sigma

    # Batched matrices: treat leading dims as batch.
    m, n = int(A.size(-2)), int(A.size(-1))
    lead_shape = A.shape[:-2]
    A_flat = A.reshape(-1, m, n)
    batch = A_flat.size(0)
    v = torch.randn(batch, n, device=A.device, dtype=A.dtype)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    for _ in range(iters):
        u = torch.bmm(A_flat, v.unsqueeze(-1)).squeeze(-1)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-8)
        v = torch.bmm(A_flat.mT, u.unsqueeze(-1)).squeeze(-1)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    sigma = torch.bmm(A_flat, v.unsqueeze(-1)).squeeze(-1).norm(dim=1)
    return sigma.view(*lead_shape)


def estimate_stable_rank(A: Tensor) -> float:
    """
    st(A) ≈ ||A||_F^2 / sigma_max(A)^2
    """
    A = A.float()
    frob_sq = (A * A).sum()
    sigma_max = power_iteration_smax(A, iters=2)
    st = frob_sq / (sigma_max**2 + 1e-8)
    return float(st)


def estimate_nuclear_rank(G: Tensor, k: int = 8) -> float:
    """
    nr(G) ≈ ||G||_*^2 / ||G||_F^2 using top-k singular values.
    """
    G = G.float()
    if G.ndim != 2:
        # Merge all leading dims into rows.
        G = G.flatten(0, -2)
    frob_sq = (G * G).sum()
    if frob_sq <= 0:
        return 0.0
    try:
        _, S, _ = torch.linalg.svd(G, full_matrices=False)
    except RuntimeError:
        # Fallback to CPU if GPU SVD fails
        _, S, _ = torch.linalg.svd(G.cpu(), full_matrices=False)
        S = S.to(G.device)
    S_top = S[:k]
    nuc_approx = S_top.sum()
    nr = (nuc_approx**2) / (frob_sq + 1e-8)
    return float(nr)


def blend_from_R(R_ema: float, threshold: float, margin: float) -> float:
    """
    Map geometry statistic R_ema to α ∈ [0,1] for blending.
    """
    if margin <= 0:
        return 1.0 if R_ema > threshold else 0.0
    if R_ema <= threshold:
        return 0.0
    if R_ema >= threshold + margin:
        return 1.0
    return float((R_ema - threshold) / margin)


def soft_threshold_matrix(M: Tensor, q: float = 0.9) -> Tuple[Tensor, Tensor]:
    """
    ROOT-style soft-thresholding: elementwise soft-threshold at quantile q.
    Returns:
        B: thresholded matrix
        tau: scalar tensor threshold
    """
    if M.numel() == 0:
        tau = torch.tensor(0.0, device=M.device, dtype=M.dtype)
        return M, tau
    flat = M.abs().reshape(-1)
    if hasattr(torch, "quantile"):
        tau = torch.quantile(flat, q)
    else:  # pragma: no cover
        k = max(int(q * flat.numel()) - 1, 0)
        tau = flat.kthvalue(k).values
    B = torch.sign(M) * torch.clamp(M.abs() - tau, min=0.0)
    return B, tau


# ---- Turbo / AdaNewton-style polar approximations (simplified) ----

# Example AdaNewton-like coefficient table for common shapes.
# For unseen shapes, we fall back to Muon's coefficients.
_ADANEWTON_COEFF_TABLE: Dict[Tuple[int, int], Tuple[float, float, float]] = {
    (2048, 2048): (3.3, -4.6, 2.0),
    (4096, 4096): (3.37, -4.9, 2.31),
    (3072, 2048): (2.9, -3.8, 1.86),
    (4096, 2048): (2.78, -3.49, 1.70),
}


def _get_adanewton_coeffs(m: int, n: int) -> Tuple[float, float, float]:
    if (m, n) in _ADANEWTON_COEFF_TABLE:
        return _ADANEWTON_COEFF_TABLE[(m, n)]
    if (n, m) in _ADANEWTON_COEFF_TABLE:
        return _ADANEWTON_COEFF_TABLE[(n, m)]
    # Default Muon-like coefficients
    return (3.44, -4.78, 2.03)


def _turbo_muon_polar(B: Tensor, ns_iters: int = 4) -> Tensor:
    """
    Turbo-Muon-inspired polar approximation:
    - Column-wise RMS preconditioning + few NS steps.
    """
    X = B.float()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    # Column-wise RMS preconditioning (per matrix, no batch mixing)
    col_rms = X.pow(2).mean(dim=-2, keepdim=True).sqrt().clamp_min(1e-6)
    X = X / col_rms

    # Normalize spectral norm
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    m, n = int(X.size(-2)), int(X.size(-1))
    a, b, c = _get_adanewton_coeffs(m, n)
    for _ in range(ns_iters):
        A = X @ X.mT
        X = a * X + b * (A @ X) + c * (A @ (A @ X))

    if transposed:
        X = X.mT
    return X


def _adanewton_polar(B: Tensor, ns_iters: int = 4) -> Tensor:
    """
    AdaNewton-style polar approximation with dimension-aware coefficients.
    """
    X = B.float()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    m, n = int(X.size(-2)), int(X.size(-1))
    a, b, c = _get_adanewton_coeffs(m, n)
    for _ in range(ns_iters):
        A = X @ X.mT
        X = a * X + b * (A @ X) + c * (A @ (A @ X))

    if transposed:
        X = X.mT
    return X


def approx_polar(B: Tensor, backend: str = "baseline", ns_iters: int = 4) -> Tensor:
    """
    Approximate polar factor U for B ≈ U H with ||U||_op ≈ 1.
    backend:
      - "baseline": Muon zeropower_via_newtonschulz5
      - "turbo":    Turbo-Muon-style preconditioned NS
      - "adanewton": AdaNewton-style NS
    """
    if backend == "turbo":
        U = _turbo_muon_polar(B, ns_iters=ns_iters)
    elif backend == "adanewton":
        U = _adanewton_polar(B, ns_iters=ns_iters)
    elif backend == "baseline":
        U = zeropower_via_newtonschulz5(B)
    else:
        raise ValueError(f"Unknown polar backend: {backend}")

    U = U.float()
    sigma_max = power_iteration_smax(U, iters=1)
    U = U / (sigma_max[..., None, None] + 1e-8)
    return U


def nor_muon_update(
    B: Tensor,
    state: Dict[str, Any],
    lr_spec_base: float,
    beta2: float = 0.99,
    eps: float = 1e-8,
    backend: str = "baseline",
    ns_iters: int = 4,
    c_rms: float = 0.2,
) -> Tensor:
    """
    NorMuon-style spectral direction for a single 2D block.

    Returns:
        spec_dir: float32 matrix step (negative update direction).
    """
    U = approx_polar(B, backend=backend, ns_iters=ns_iters)
    m, n = int(U.size(-2)), int(U.size(-1))
    lead_shape = U.shape[:-2]

    v_row = state.get("v_row")
    if v_row is None or tuple(v_row.shape) != tuple(lead_shape + (m,)):
        v_row = torch.zeros(*lead_shape, m, device=U.device, dtype=torch.float32)

    row_ms = U.pow(2).mean(dim=-1)  # [..., m]
    v_row = beta2 * v_row + (1.0 - beta2) * row_ms
    state["v_row"] = v_row

    denom = (v_row + eps).sqrt().unsqueeze(-1)  # [..., m, 1]
    O_hat = U / denom

    frob = O_hat.pow(2).sum(dim=(-2, -1)).sqrt()
    rms = frob / math.sqrt(float(m * n))
    eta_eff = c_rms * lr_spec_base / (rms + eps)

    spec_dir = -eta_eff[..., None, None] * O_hat
    return spec_dir


def muon_like_spectral_update(
    B: Tensor,
    lr_spec_base: float,
    backend: str = "baseline",
    ns_iters: int = 4,
) -> Tensor:
    """
    Plain Muon-style spectral update (no NorMuon row-wise scaling).
    """
    U = approx_polar(B, backend=backend, ns_iters=ns_iters)
    m, n = int(U.size(-2)), int(U.size(-1))
    shape_scale = math.sqrt(max(1.0, float(m) / max(1.0, float(n))))
    return -lr_spec_base * shape_scale * U


# ---- AdamW branches ----

def _adamw_update_param(
    p: Tensor,
    state: Dict[str, Any],
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    """
    Full AdamW update for non-spectral (AdamW-only) parameters.
    """
    if p.grad is None:
        return
    g = p.grad
    if g.is_sparse:  # pragma: no cover
        raise RuntimeError("NeoMuon does not support sparse gradients")

    g32 = g.detach().to(torch.float32)

    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg is None:
        exp_avg = torch.zeros_like(p, dtype=torch.float32)
        exp_avg_sq = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq
        state["adam_step"] = 0

    beta1, beta2 = betas
    t = int(state.get("adam_step", 0)) + 1
    state["adam_step"] = t

    exp_avg.mul_(beta1).add_(g32, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)

    bias_correction1 = 1.0 - beta1**t
    bias_correction2 = 1.0 - beta2**t

    m_hat = exp_avg / bias_correction1
    v_hat = exp_avg_sq / bias_correction2

    denom = v_hat.sqrt().add_(eps)
    step_dir = -lr * (m_hat / denom)

    # Decoupled weight decay
    if weight_decay != 0.0:
        p.mul_(1.0 - lr * weight_decay)

    p.add_(step_dir.to(p.dtype))


def _adamw_direction_spectral(
    p: Tensor,
    g32: Tensor,
    state: Dict[str, Any],
    lr: float,
    betas: Tuple[float, float],
    eps: float,
) -> Tensor:
    """
    AdamW-like direction for spectral parameters (no weight decay applied here).
    Returns float32 direction to blend with spectral branch.
    """
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg is None:
        exp_avg = torch.zeros_like(p, dtype=torch.float32)
        exp_avg_sq = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq
        state["adam_step"] = 0

    beta1, beta2 = betas
    t = int(state.get("adam_step", 0)) + 1
    state["adam_step"] = t

    exp_avg.mul_(beta1).add_(g32, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)

    bias_correction1 = 1.0 - beta1**t
    bias_correction2 = 1.0 - beta2**t

    m_hat = exp_avg / bias_correction1
    v_hat = exp_avg_sq / bias_correction2

    denom = v_hat.sqrt().add_(eps)
    step_dir = -lr * (m_hat / denom)
    return step_dir  # float32


# ---- NeoMuon Optimizer ----

class NeoMuon(Optimizer):
    """
    NeoMuon: geometry-aware hybrid optimizer built on Muon.

    Param groups:
      - spectral=True  (default): Muon / NeoMuon spectral blocks (2D bfloat16).
      - spectral=False: AdamW-only parameters (embeddings, heads, biases, norms, etc).
    
    Spectral branch LR:
      - lr_spec controls the spectral update strength for spectral groups.
      - If omitted, lr_spec defaults to lr for that group.
      - For a "hotter" spectral branch (common in prior NeoMuon/Muon recipes),
        pass lr_spec explicitly, e.g. 5–10× lr.

    Core behaviour:
      - If ALL tweaks are disabled:
          enable_spectral_gating=False,
          enable_initial_adamw_steps=False,
          enable_normuon=False,
          enable_turbomuon=False,
          enable_root=False,
        then spectral groups are updated EXACTLY like the reference Muon
        (same kernel and distributed pattern). AdamW-only groups still use
        AdamW.

      - When tweaks are enabled:
          * AdamW-only groups: pure AdamW.
          * Spectral groups: AdamW baseline + optional spectral branch
            (NorMuon + Turbo/AdaNewton + ROOT soft-threshold), gated by
            nr(G)/st(A), with an AdamW warmup period if requested.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[Dict[str, Any]],
        # AdamW / Euclidean hyperparams
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        # Muon spectral hyperparams
        muon_momentum: float = 0.95,
        lr_spec: Optional[float] = None,
        # Scheduling / gating
        initial_adamw_steps: int = 3,
        spec_interval: int = 4,
        reestimate_interval: int = 50,
        nr_st_threshold: float = 1.5,
        blend_margin: float = 1.0,
        ema_rho: float = 0.1,
        soft_q: float = 0.9,
        ns_iters: int = 4,
        rank: int = 0,
        world_size: int = 1,
        # Tweak toggles
        enable_spectral_gating: bool = True,
        enable_initial_adamw_steps: bool = True,
        enable_normuon: bool = True,
        enable_turbomuon: bool = True,
        enable_root: bool = True,
    ) -> None:
        if lr_spec is None:
            # By default, keep the spectral branch on the same LR as the Euclidean LR.
            # If you want a "hotter" spectral branch (often 5–10×), pass lr_spec explicitly.
            lr_spec = lr

        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            lr_spec=lr_spec,
            spectral=True,  # default; can be overridden per param-group
        )
        super().__init__(params, defaults)

        self.rank = int(rank)
        self.world_size = int(world_size)

        # Feature toggles
        self.enable_spectral_gating = bool(enable_spectral_gating)
        self.enable_initial_adamw_steps = bool(enable_initial_adamw_steps)
        self.enable_normuon = bool(enable_normuon)
        self.enable_turbomuon = bool(enable_turbomuon)
        self.enable_root = bool(enable_root)

        # Geometry / scheduling
        self.initial_adamw_steps = int(initial_adamw_steps)
        self.spec_interval = int(spec_interval)
        self.reestimate_interval = int(reestimate_interval)
        self.nr_st_threshold = float(nr_st_threshold)
        self.blend_margin = float(blend_margin)
        self.ema_rho = float(ema_rho)
        self.soft_q = float(soft_q)
        self.ns_iters = int(ns_iters)
        self.alpha_max = 1.0

        self._step_count: int = 0

        # Precompute whether we are in "pure Muon" mode for spectral groups.
        self._compat_mode = not (
            self.enable_spectral_gating
            or self.enable_initial_adamw_steps
            or self.enable_normuon
            or self.enable_turbomuon
            or self.enable_root
        )

        # Muon constraint: spectral parameters must be bfloat16
        for group in self.param_groups:
            if group.get("spectral", True):
                for p in group["params"]:
                    if not isinstance(p, Tensor):
                        continue
                    # Only enforce for >=2D tensors; 0/1D will usually go to AdamW or be ignored.
                    if p.ndim >= 2 and p.dtype != torch.bfloat16:
                        raise ValueError(
                            "NeoMuon spectral parameters (2D) must be torch.bfloat16 "
                            f"(got dtype={p.dtype} for param shape {tuple(p.shape)}). "
                            "Put non-bfloat16 or non-2D parameters into a param group "
                            "with spectral=False to use pure AdamW."
                        )

    @torch.no_grad()
    def set_last_activation(self, p: Tensor, activation: Tensor) -> None:
        """
        Attach incoming activation A_{l-1} for a spectral weight matrix p.
        Used for nr/st-based spectral gating when enabled.
        """
        state = self.state[p]
        state["last_activation"] = activation.detach()

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        global_step = self._step_count

        # 1) AdamW-only param groups (spectral=False) – always AdamW, independent of tweaks.
        for group in self.param_groups:
            if group.get("spectral", True):
                continue  # handled in spectral pass

            lr: float = float(group["lr"])
            betas: Tuple[float, float] = tuple(group["betas"])  # type: ignore[arg-type]
            eps: float = float(group["eps"])
            weight_decay: float = float(group["weight_decay"])

            for p in group["params"]:
                if not isinstance(p, Tensor):
                    continue
                if p.grad is None:
                    continue
                state = self.state[p]
                _adamw_update_param(p, state, lr, betas, eps, weight_decay)

        # If no spectral groups, we're done.
        has_spectral = any(group.get("spectral", True) for group in self.param_groups)
        if not has_spectral:
            return loss

        use_dist = (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )

        # 2) Spectral groups.
        if self._compat_mode:
            # --- Pure Muon mode for spectral groups (reference behaviour) ---
            futures: List[torch.futures.Future] = []

            for group in self.param_groups:
                if not group.get("spectral", True):
                    continue

                params: List[Tensor] = [p for p in group["params"] if isinstance(p, Tensor)]
                if not params:
                    continue

                if use_dist:
                    params_pad = params + [torch.empty_like(params[-1])] * self.world_size
                    momentum_t: Optional[Tensor] = None

                    for base_i in range(0, len(params), self.world_size):
                        idx = base_i + self.rank
                        if idx < len(params):
                            p = params[idx]
                            if p.grad is not None:
                                state = self.state[p]
                                if "mantissa" not in state:
                                    state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                                if momentum_t is None:
                                    momentum_t = _as_full_prec_tensor(
                                        float(group["momentum"]), device=p.device
                                    )

                                eff_lr = float(group["lr"]) * math.sqrt(
                                    max(
                                        1.0,
                                        float(p.size(-2))
                                        / max(1.0, float(p.size(-1))),
                                    )
                                )
                                eff_lr_t = _as_full_prec_tensor(eff_lr, device=p.device)

                                eff_wd = float(group["lr"]) * float(group["weight_decay"]) * float(
                                    getattr(p, "wd_mul", 1.0)
                                )
                                eff_wd_t = _as_full_prec_tensor(eff_wd, device=p.device)

                                _muon_update_kernel(
                                    p.view(torch.uint16),
                                    state["mantissa"],
                                    state["momentum_buffer"],
                                    p.grad,
                                    momentum_t,
                                    eff_lr_t,
                                    eff_wd_t,
                                )
                            src = params_pad[idx]
                        else:
                            src = params_pad[-1]

                        out_list = params_pad[base_i : base_i + self.world_size]
                        work = dist.all_gather(out_list, src, async_op=True)
                        futures.append(work.get_future())

                    torch.futures.collect_all(futures).wait()

                else:
                    momentum_t: Optional[Tensor] = None
                    for p in params:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        if "mantissa" not in state:
                            state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                        if momentum_t is None:
                            momentum_t = _as_full_prec_tensor(
                                float(group["momentum"]), device=p.device
                            )

                        eff_lr = float(group["lr"]) * math.sqrt(
                            max(
                                1.0,
                                float(p.size(-2)) / max(1.0, float(p.size(-1))),
                            )
                        )
                        eff_lr_t = _as_full_prec_tensor(eff_lr, device=p.device)

                        eff_wd = float(group["lr"]) * float(group["weight_decay"]) * float(
                            getattr(p, "wd_mul", 1.0)
                        )
                        eff_wd_t = _as_full_prec_tensor(eff_wd, device=p.device)

                        _muon_update_kernel(
                            p.view(torch.uint16),
                            state["mantissa"],
                            state["momentum_buffer"],
                            p.grad,
                            momentum_t,
                            eff_lr_t,
                            eff_wd_t,
                        )

            return loss

        # --- NeoMuon mode for spectral groups (tweaks enabled) ---

        warmup_phase = (
            self.enable_initial_adamw_steps
            and self.initial_adamw_steps > 0
            and global_step <= self.initial_adamw_steps
        )

        for group in self.param_groups:
            if not group.get("spectral", True):
                continue

            params: List[Tensor] = [p for p in group["params"] if isinstance(p, Tensor)]
            if not params:
                continue

            lr: float = float(group["lr"])
            betas: Tuple[float, float] = tuple(group["betas"])  # type: ignore[arg-type]
            eps: float = float(group["eps"])
            weight_decay: float = float(group["weight_decay"])
            lr_spec: float = float(group.get("lr_spec", lr))

            if use_dist:
                params_pad = params + [torch.empty_like(params[-1])] * self.world_size

                for base_i in range(0, len(params), self.world_size):
                    idx = base_i + self.rank
                    if idx < len(params):
                        p = params[idx]
                        if p.grad is not None:
                            self._update_spectral_param_neomuon(
                                p=p,
                                group_lr=lr,
                                lr_spec=lr_spec,
                                betas=betas,
                                eps=eps,
                                weight_decay=weight_decay,
                                global_step=global_step,
                                warmup_phase=warmup_phase,
                            )
                        src = params_pad[idx]
                    else:
                        src = params_pad[-1]

                    out_list = params_pad[base_i : base_i + self.world_size]
                    dist.all_gather(out_list, src, async_op=False)

            else:
                for p in params:
                    if p.grad is None:
                        continue
                    self._update_spectral_param_neomuon(
                        p=p,
                        group_lr=lr,
                        lr_spec=lr_spec,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        global_step=global_step,
                        warmup_phase=warmup_phase,
                    )

        return loss

    @torch.no_grad()
    def _update_spectral_param_neomuon(
        self,
        p: Tensor,
        group_lr: float,
        lr_spec: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        global_step: int,
        warmup_phase: bool,
    ) -> None:
        """
        NeoMuon update for a single spectral parameter (2D matrix).
        0/1D tensors in spectral groups fall back to AdamW.
        """
        state = self.state[p]
        g = p.grad
        if g is None:
            return

        # 0/1D in spectral groups → treat as AdamW for robustness.
        if p.ndim < 2:
            _adamw_update_param(p, state, group_lr, betas, eps, weight_decay)
            return

        # Convert gradient once (shared by AdamW + spectral branch).
        g32 = g.detach().to(torch.float32)

        # AdamW baseline direction (no weight decay).
        adam_dir = _adamw_direction_spectral(p, g32, state, group_lr, betas, eps)

        # Warmup: just AdamW for a few steps (no spectral machinery).
        if warmup_phase:
            if weight_decay != 0.0:
                p.mul_(1.0 - group_lr * weight_decay)
            p.add_(adam_dir.to(p.dtype))
            return

        # Geometry / gating stats (nr/st) used only when requested.
        R_ema: float = float(state.get("R_ema", 1.0))
        if (
            self.enable_spectral_gating
            and self.reestimate_interval > 0
            and global_step % self.reestimate_interval == 0
            and ("last_activation" in state)
        ):
            try:
                A = state["last_activation"]
                st = estimate_stable_rank(A)
                nr = estimate_nuclear_rank(g32)
                R = nr / max(st, 1e-6)
                R_ema = (1.0 - self.ema_rho) * R_ema + self.ema_rho * R
                state["R_ema"] = R_ema
            except Exception:
                # If estimation fails, just keep previous R_ema.
                pass

        # Decide whether to use spectral branch and blending coefficient α.
        if not self.enable_spectral_gating:
            use_spectral = True
            alpha = 1.0
        else:
            if (self.spec_interval <= 0) or (global_step % self.spec_interval != 0):
                use_spectral = False
                alpha = 0.0
            elif R_ema <= self.nr_st_threshold:
                use_spectral = False
                alpha = 0.0
            else:
                alpha = blend_from_R(R_ema, self.nr_st_threshold, self.blend_margin)
                alpha = float(max(0.0, min(self.alpha_max, alpha)))
                use_spectral = alpha > 0.0

        if not use_spectral:
            # Geometry not favourable or off-step: pure AdamW.
            if weight_decay != 0.0:
                p.mul_(1.0 - group_lr * weight_decay)
            p.add_(adam_dir.to(p.dtype))
            return

        # --- Spectral branch (NorMuon / Muon-like) ---

        # Matrix momentum M for spectral branch.
        M = state.get("M")
        if M is None:
            M = torch.zeros_like(p, dtype=torch.float32)
        beta1_spec = betas[0]
        M.mul_(beta1_spec).add_(g32, alpha=1.0 - beta1_spec)
        state["M"] = M

        # ROOT-style soft-thresholding
        if self.enable_root:
            B, tau = soft_threshold_matrix(M, q=self.soft_q)
            state["root_tau"] = tau
        else:
            B = M

        # Choose polar backend
        if self.enable_turbomuon:
            backend = "turbo"
        elif self.enable_root:
            backend = "adanewton"
        else:
            backend = "baseline"

        # NorMuon vs plain Muon-like spectral direction
        if self.enable_normuon:
            spec_dir = nor_muon_update(
                B,
                state=state,
                lr_spec_base=lr_spec,
                beta2=betas[1],
                eps=eps,
                backend=backend,
                ns_iters=self.ns_iters,
                c_rms=0.2,
            )
        else:
            spec_dir = muon_like_spectral_update(
                B,
                lr_spec_base=lr_spec,
                backend=backend,
                ns_iters=self.ns_iters,
            )

        # Blend AdamW and spectral directions.
        total_dir = (1.0 - alpha) * adam_dir + alpha * spec_dir

        # Decoupled weight decay (same as AdamW-only).
        if weight_decay != 0.0:
            p.mul_(1.0 - group_lr * weight_decay)

        p.add_(total_dir.to(p.dtype))
