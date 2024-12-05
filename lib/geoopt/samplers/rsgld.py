import math

import torch

from lib.geoopt.tensor import ManifoldParameter, ManifoldTensor
from lib.geoopt.samplers.base import Sampler

__all__ = ["RSGLD"]


class RSGLD(Sampler):
    r"""
    Riemannian Stochastic Gradient Langevin Dynamics.

    Parameters
    ----------
    params : iterable
        iterables of tensors for which to perform sampling
    epsilon : float
        step size
    """

    def __init__(self, params, epsilon=1e-3):
        defaults = dict(epsilon=epsilon)
        super().__init__(params, defaults)

    def step(self, closure):
        logp = closure()
        logp.backward()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                        manifold = p.manifold
                    else:
                        manifold = self._default_manifold

                    egrad2rgrad, retr = manifold.egrad2rgrad, manifold.retr
                    epsilon = group["epsilon"]

                    n = torch.randn_like(p).mul_(math.sqrt(epsilon))
                    r = egrad2rgrad(p, 0.5 * epsilon * p.grad + n)
                    # use copy only for user facing point
                    p.copy_(retr(p, r))
                    p.grad.zero_()

        if not self.burnin:
            self.steps += 1
            self.log_probs.append(logp.item())

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            p.copy_(p.manifold.projx(p))
