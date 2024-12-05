from ..manifolds import Euclidean
import torch


class OptimMixin(object):
    _default_manifold = Euclidean()

    def __init__(self, *args, stabilize=None, **kwargs):
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def add_param_group(self, param_group: dict):
        param_group.setdefault("stabilize", self._stabilize)
        return super().add_param_group(param_group)

    def stabilize_group(self, group):
        pass

    def stabilize(self):
        """Stabilize parameters if they are off-manifold due to numerical reasons."""
        for group in self.param_groups:
            self.stabilize_group(group)


class SparseMixin(object):
    def add_param_group(self, param_group):
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but "
                "the ordering of tensors in sets will change between runs. Please use a list instead."
            )
        else:
            param_group["params"] = list(params)
        for param in param_group["params"]:
            if param.dim() != 2:
                raise ValueError(
                    "Param for sparse optimizer should be matrix valued, but got shape {}".format(
                        param.shape
                    )
                )
        return super().add_param_group(param_group)
