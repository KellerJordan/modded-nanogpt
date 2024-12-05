from torch import optim as optim

from lib.geoopt.optim.mixin import OptimMixin
from lib.geoopt.tensor import ManifoldParameter, ManifoldTensor


__all__ = ["Sampler"]


class Sampler(OptimMixin, optim.Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        self.n_rejected = 0
        self.steps = 0
        self.burnin = True

        self.log_probs = []
        self.acceptance_probs = []
        for group in self.param_groups:
            for p in group["params"]:
                if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    if not p.manifold.reversible:
                        raise ValueError(
                            "Sampling methods can't me applied to manifolds that "
                            "do not implement reversible retraction"
                        )

    @property
    def rejection_rate(self):
        if self.steps > 0:
            return self.n_rejected / self.steps
        else:
            return 0.0

    def step(self, closure):
        """
        Perform a single sampling step.

        Arguments
        ---------
        closure: callable
            A closure that reevaluates the model
            and returns the log probability.
        """
        raise NotImplementedError
