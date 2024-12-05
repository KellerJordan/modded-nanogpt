import torch.optim.optimizer
from ..tensor import ManifoldParameter, ManifoldTensor
from .mixin import OptimMixin, SparseMixin

__all__ = ["SparseRiemannianSGD"]


class SparseRiemannianSGD(OptimMixin, SparseMixin, torch.optim.Optimizer):
    r"""
    Implements lazy version of SGD algorithm suitable for sparse gradients.

    In this variant, only moments that show up in the gradient get updated, and
    only those portions of the gradient get applied to the parameters.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float
        learning rate
    momentum : float (optional)
        momentum factor (default: 0)
    dampening : float (optional)
        dampening for momentum (default: 0)
    nesterov : bool (optional)
        enables Nesterov momentum (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        nesterov=False,
        stabilize=None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults, stabilize=stabilize)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                learning_rate = group["lr"]
                group["step"] += 1
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if not grad.is_sparse:
                        raise RuntimeError(
                            "SparseRiemannianAdam does not support sparse gradients, use RiemannianAdam instead"
                        )
                    # select rows that contain gradient
                    rows = grad.coalesce().indices()[0].unique()
                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = grad.to_dense().clone()
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    full_point = point
                    # only nonzero rows are required to make an update
                    grad = grad.index_select(0, rows).to_dense()
                    point = point[rows]

                    grad = manifold.egrad2rgrad(point, grad)
                    if momentum > 0:
                        momentum_buffer = state["momentum_buffer"][rows]
                        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
                        if nesterov:
                            grad = grad.add_(momentum_buffer, alpha=momentum)
                        else:
                            grad = momentum_buffer
                        # we have all the things projected
                        new_point, new_momentum_buffer = manifold.retr_transp(
                            point, -learning_rate * grad, momentum_buffer
                        )
                        # use copy only for user facing point
                        state["momentum_buffer"][rows] = new_momentum_buffer
                        full_point[rows] = new_point
                    else:
                        new_point = manifold.retr(point, -learning_rate * grad)
                        full_point[rows] = new_point

                if (
                    group["stabilize"] is not None
                    and group["step"] % group["stabilize"] == 0
                ):
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            manifold = p.manifold
            momentum = group["momentum"]
            p.copy_(manifold.projx(p))
            if momentum > 0:
                param_state = self.state[p]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(manifold.proju(p, buf))
