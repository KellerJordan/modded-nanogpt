import torch
import torch.distributed as dist
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ∈ [1 - l, 1 + r], which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)

    acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
    acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
    acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    mantissa.copy_(acc_m_u32.to(torch.uint16))


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        assert all(p.dtype == torch.bfloat16 for group in self.param_groups for p in group["params"])

    @torch.no_grad()
    def step(self):
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = torch._as_tensor_fullprec(group["momentum"])
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    grad = p.grad
                    if grad is not None:
                        update(
                            p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                            grad, momentum,
                            eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                            eff_weight_decay=torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                        )
                futures.append(dist.all_gather(params_pad[base_i:base_i + self.world_size], params_pad[base_i + self.rank], async_op=True).get_future())
        torch.futures.collect_all(futures).wait()
