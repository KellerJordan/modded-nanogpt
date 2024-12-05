from typing import Union, Tuple, Optional
import torch
from .base import Manifold, ScalingInfo
from ..utils import size2shape, broadcast_shapes
import lib.geoopt as geoopt


__all__ = ["Euclidean"]


class Euclidean(Manifold):
    """
    Simple Euclidean manifold, every coordinate is treated as an independent element.

    Parameters
    ----------
    ndim : int
        number of trailing dimensions treated as manifold dimensions. All the operations acting on cuch
        as inner products, etc will respect the :attr:`ndim`.
    """

    __scaling__ = Manifold.__scaling__.copy()
    name = "Euclidean"
    ndim = 0
    reversible = True

    def __init__(self, ndim=0):
        super().__init__()
        self.ndim = ndim

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        return True, None

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v
        if self.ndim > 0:
            inner = inner.sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
            x_shape = x.shape[: -self.ndim] + (1,) * self.ndim * keepdim
        else:
            x_shape = x.shape
        i_shape = inner.shape
        target_shape = broadcast_shapes(x_shape, i_shape)
        return inner.expand(target_shape)

    def component_inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None
    ) -> torch.Tensor:
        # it is possible to factorize the manifold
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v
        target_shape = broadcast_shapes(x.shape, inner.shape)
        return inner.expand(target_shape)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False):
        if self.ndim > 0:
            return u.norm(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return u.abs()

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - x

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        if self.ndim > 0:
            return (x - y).norm(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return (x - y).abs()

    def dist2(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        if self.ndim > 0:
            return (x - y).pow(2).sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return (x - y).pow(2)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, y.shape, v.shape)
        return v.expand(target_shape)

    @__scaling__(ScalingInfo(std=-1), "random")
    def random_normal(
        self, *size, mean=0.0, std=1.0, device=None, dtype=None
    ) -> "geoopt.ManifoldTensor":
        """
        Create a point on the manifold, measure is induced by Normal distribution.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype

        Returns
        -------
        ManifoldTensor
            random point on the manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        mean = torch.as_tensor(mean, device=device, dtype=dtype)
        std = torch.as_tensor(std, device=device, dtype=dtype)
        tens = std.new_empty(*size).normal_() * std + mean
        return geoopt.ManifoldTensor(tens, manifold=self)

    random = random_normal

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
        """
        self._assert_check_shape(size2shape(*size), "x")
        return geoopt.ManifoldTensor(
            torch.zeros(*size, dtype=dtype, device=device), manifold=self
        )

    def extra_repr(self):
        return "ndim={}".format(self.ndim)
