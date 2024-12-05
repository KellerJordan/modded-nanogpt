import torch.optim

from .mixin import OptimMixin, SparseMixin
from ..tensor import ManifoldParameter, ManifoldTensor


__all__ = ["SparseRiemannianAdam"]


class SparseRiemannianAdam(OptimMixin, SparseMixin, torch.optim.Optimizer):
    r"""
    Implements lazy version of Adam algorithm suitable for sparse gradients.

    In this variant, only moments that show up in the gradient get updated, and
    only those portions of the gradient get applied to the parameters.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float (optional)
        learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps : float (optional)
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    amsgrad : bool (optional)
        whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)


    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)
        super(SparseRiemannianAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SparseRiemannianAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                betas = group["betas"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                stablilize = False
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    if not grad.is_sparse:
                        raise RuntimeError(
                            "SparseRiemannianAdam does not support sparse gradients, use RiemannianAdam instead"
                        )
                    rows = grad.coalesce().indices()[0].unique()
                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    state["step"] += 1
                    full_point = point
                    # only nonzero rows are required to make an update
                    grad = grad.index_select(0, rows).to_dense()
                    # this takes not view, but copy, we are required to make updates later
                    point = point[rows]
                    exp_avg = state["exp_avg"][rows]
                    exp_avg_sq = state["exp_avg_sq"][rows]
                    # actual step
                    grad = manifold.egrad2rgrad(point, grad)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(
                        manifold.component_inner(point, grad), alpha=1 - betas[1]
                    )
                    bias_correction1 = 1 - betas[0] ** state["step"]
                    bias_correction2 = 1 - betas[1] ** state["step"]
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"][rows]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        max_exp_avg_sq.div_(bias_correction2).sqrt_()
                        state["max_exp_avg_sq"][rows] = max_exp_avg_sq
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg.div(bias_correction1) / denom.add_(eps)
                    # transport the exponential averaging to the new point
                    new_point, exp_avg_new = manifold.retr_transp(
                        point, -learning_rate * direction, exp_avg
                    )
                    # now we update all full tensors
                    full_point[rows] = new_point
                    state["exp_avg"][rows] = exp_avg_new
                    state["exp_avg_sq"][rows] = exp_avg_sq

                    if (
                        group["stabilize"] is not None
                        and state["step"] % group["stabilize"] == 0
                    ):
                        stablilize = True
                if stablilize:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            exp_avg = state["exp_avg"]
            p.copy_(manifold.projx(p))
            exp_avg.copy_(manifold.proju(p, exp_avg))
