import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer

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


def power_iteration_smax(
    A: Tensor,
    iters: int = 2,
    generator: Optional[torch.Generator] = None,
    v_buf: Optional[Tensor] = None,
) -> Tensor:
    """
    Approximate largest singular value of a (batched) 2D tensor via power iteration.
    """
    A = A.float()
    if A.ndim == 2:
        m, n = A.shape
        if (
            v_buf is not None
            and (v_buf.shape != (n,) or v_buf.device != A.device or v_buf.dtype != A.dtype)
        ):
            v_buf = None

        if v_buf is None:
            v = torch.randn(n, device=A.device, dtype=A.dtype, generator=generator)
        else:
            v = v_buf

        v.div_(v.norm() + 1e-8)
        for _ in range(iters):
            u = A @ v
            u = u / (u.norm() + 1e-8)
            v_new = A.mT @ u
            v_new = v_new / (v_new.norm() + 1e-8)
            if v_buf is None:
                v = v_new
            else:
                v.copy_(v_new)
        sigma = (A @ v).norm()
        return sigma

    # Batched matrices: treat leading dims as batch.
    m, n = int(A.size(-2)), int(A.size(-1))
    lead_shape = A.shape[:-2]
    A_flat = A.reshape(-1, m, n)
    batch = A_flat.size(0)

    if (
        v_buf is not None
        and (
            v_buf.shape != (*lead_shape, n)
            or v_buf.device != A.device
            or v_buf.dtype != A.dtype
        )
    ):
        v_buf = None

    if v_buf is None:
        v = torch.randn(batch, n, device=A.device, dtype=A.dtype, generator=generator)
    else:
        v = v_buf.reshape(batch, n)

    v.div_(v.norm(dim=1, keepdim=True) + 1e-8)
    for _ in range(iters):
        u = torch.bmm(A_flat, v.unsqueeze(-1)).squeeze(-1)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-8)
        v_new = torch.bmm(A_flat.mT, u.unsqueeze(-1)).squeeze(-1)
        v_new = v_new / (v_new.norm(dim=1, keepdim=True) + 1e-8)
        if v_buf is None:
            v = v_new
        else:
            v.copy_(v_new)
    sigma = torch.bmm(A_flat, v.unsqueeze(-1)).squeeze(-1).norm(dim=1)
    return sigma.view(*lead_shape)


# ---- Turbo-Muon-style polar approximations ----

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


def approx_polar(
    B: Tensor,
    backend: str = "baseline",
    ns_iters: int = 4,
    generator: Optional[torch.Generator] = None,
    v_buf: Optional[Tensor] = None,
) -> Tensor:
    """
    Approximate polar factor U for B ≈ U H with ||U||_op ≈ 1.
    backend:
      - "baseline": Muon zeropower_via_newtonschulz5
      - "turbo":    Turbo-Muon-style preconditioned NS
    """
    if backend == "turbo":
        U = _turbo_muon_polar(B, ns_iters=ns_iters)
    elif backend == "baseline":
        U = zeropower_via_newtonschulz5(B)
    else:
        raise ValueError(f"Unknown polar backend: {backend}")

    U = U.float()
    sigma_max = power_iteration_smax(U, iters=1, generator=generator, v_buf=v_buf)
    U = U / (sigma_max[..., None, None] + 1e-8)
    return U


def muon_like_spectral_update(
    B: Tensor,
    lr_spec_base: float,
    backend: str = "baseline",
    ns_iters: int = 4,
    generator: Optional[torch.Generator] = None,
    v_buf: Optional[Tensor] = None,
) -> Tensor:
    """
    Plain Muon-style spectral update.
    """
    U = approx_polar(B, backend=backend, ns_iters=ns_iters, generator=generator, v_buf=v_buf)
    m, n = int(U.size(-2)), int(U.size(-1))
    shape_scale = math.sqrt(max(1.0, float(m) / max(1.0, float(n))))
    return -lr_spec_base * shape_scale * U


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
        raise RuntimeError("Muon does not support sparse gradients")

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


class Muon(Optimizer):
    """
    Muon: hybrid optimizer for switch-bank.

    - Muon mode is based on Keller Jordan's Muon (MomentUm Orthogonalized by Newton–Schulz):
      https://kellerjordan.github.io/posts/muon/
    - TurboMuon mode is a faster Muon-inspired approximate-polar update (see `_turbo_muon_polar`
      and `approx_polar`).

    Param groups:
      - spectral=True  (default): 2D+ bfloat16 matrices updated via Muon (Turbo off)
        or TurboMuon-style spectral updates (Turbo on).
      - spectral=False: AdamW-only parameters (embeddings, heads, biases, norms, etc).
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
        ns_iters: int = 4,
        rank: int = 0,
        world_size: int = 1,
        enable_turbomuon: bool = True,
    ) -> None:
        if lr_spec is None:
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
        self.enable_turbomuon = bool(enable_turbomuon)
        self.ns_iters = int(ns_iters)

        self._step_count: int = 0
        self._turbo_rng: Optional[torch.Generator] = None
        self._turbo_rng_state: Optional[Tensor] = None
        self._turbomuon_warmstart_smax: bool = False

        # Muon constraint: spectral parameters must be bfloat16
        for group in self.param_groups:
            if group.get("spectral", True):
                for p in group["params"]:
                    if not isinstance(p, Tensor):
                        continue
                    # Only enforce for >=2D tensors; 0/1D will usually go to AdamW or be ignored.
                    if p.ndim >= 2 and p.dtype != torch.bfloat16:
                        raise ValueError(
                            "Muon spectral parameters (2D) must be torch.bfloat16 "
                            f"(got dtype={p.dtype} for param shape {tuple(p.shape)}). "
                            "Put non-bfloat16 or non-2D parameters into a param group "
                            "with spectral=False to use pure AdamW."
                        )

    def _get_turbo_rng(self, device: torch.device) -> torch.Generator:
        if self._turbo_rng is not None:
            return self._turbo_rng

        gen = torch.Generator(device=device)
        base_seed = int(torch.initial_seed())
        seed = (base_seed + 1000003 * int(self.rank)) & 0xFFFFFFFFFFFFFFFF
        gen.manual_seed(seed)

        if self._turbo_rng_state is not None:
            state_cpu = self._turbo_rng_state.detach().to(device="cpu")
            gen.set_state(state_cpu)
            self._turbo_rng_state = None

        self._turbo_rng = gen
        return gen

    def set_turbomuon_warmstart_smax(self, enabled: bool) -> None:
        self._turbomuon_warmstart_smax = bool(enabled)

    def state_dict(self) -> Dict[str, Any]:
        out = super().state_dict()
        if not out.get("param_groups"):
            return out

        turbo_state: Optional[Tensor] = None
        if self._turbo_rng is not None:
            turbo_state = self._turbo_rng.get_state()
        elif self._turbo_rng_state is not None:
            turbo_state = self._turbo_rng_state

        if turbo_state is not None:
            out["param_groups"][0]["turbo_rng_state"] = turbo_state.detach().to(device="cpu")

        return out

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        turbo_state: Optional[Tensor] = None
        try:
            param_groups = state_dict.get("param_groups", [])
            if param_groups:
                turbo_state = param_groups[0].get("turbo_rng_state")
        except Exception:
            turbo_state = None

        super().load_state_dict(state_dict)
        self._turbo_rng = None
        self._turbo_rng_state = turbo_state

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        global_step = self._step_count

        # 1) AdamW-only param groups (spectral=False) – always AdamW.
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
            self.world_size > 1 and dist.is_available() and dist.is_initialized()
        )

        # 2) Spectral groups.
        if not self.enable_turbomuon:
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

        # --- TurboMuon mode for spectral groups ---
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
                            self._update_spectral_param_turbomuon(
                                p=p,
                                group_lr=lr,
                                lr_spec=lr_spec,
                                betas=betas,
                                eps=eps,
                                weight_decay=weight_decay,
                                global_step=global_step,
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
                    self._update_spectral_param_turbomuon(
                        p=p,
                        group_lr=lr,
                        lr_spec=lr_spec,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        global_step=global_step,
                    )

        return loss

    @torch.no_grad()
    def _update_spectral_param_turbomuon(
        self,
        p: Tensor,
        group_lr: float,
        lr_spec: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        global_step: int,
    ) -> None:
        state = self.state[p]
        g = p.grad
        if g is None:
            return

        # 0/1D in spectral groups → treat as AdamW for robustness.
        if p.ndim < 2:
            _adamw_update_param(p, state, group_lr, betas, eps, weight_decay)
            return

        g32 = g.detach().to(torch.float32)

        M = state.get("M")
        if M is None:
            M = torch.zeros_like(p, dtype=torch.float32)
        beta1_spec = betas[0]
        M.mul_(beta1_spec).add_(g32, alpha=1.0 - beta1_spec)
        state["M"] = M

        turbo_gen = self._get_turbo_rng(device=p.device)
        v_buf = None
        if self._turbomuon_warmstart_smax:
            v_buf = state.get("pi_v")
            if v_buf is None:
                v_shape = (*p.shape[:-2], int(p.size(-1)))
                v_buf = torch.randn(v_shape, device=p.device, dtype=torch.float32, generator=turbo_gen)
                state["pi_v"] = v_buf

        spec_dir = muon_like_spectral_update(
            M,
            lr_spec_base=lr_spec,
            backend="turbo",
            ns_iters=self.ns_iters,
            generator=turbo_gen,
            v_buf=v_buf,
        )

        if weight_decay != 0.0:
            p.mul_(1.0 - group_lr * weight_decay)

        p.add_(spec_dir.to(p.dtype))
