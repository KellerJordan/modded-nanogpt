import torch as th
import torch.nn
import numpy as np
from typing import Tuple, Optional
from . import math
import lib.geoopt as geoopt
from ..base import Manifold, ScalingInfo
from ...utils import size2shape, broadcast_shapes

__all__ = ["Lorentz"]

_lorentz_ball_doc = r"""
    Lorentz model

    Parameters
    ----------
    k : float|tensor
        manifold negative curvature

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision
"""


class Lorentz(Manifold):
    __doc__ = r"""{}
    """.format(
        _lorentz_ball_doc
    )

    ndim = 1
    reversible = False
    name = "Lorentz"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, k=1.0, learnable=False):
        super().__init__()
        k = torch.as_tensor(k)
        if not torch.is_floating_point(k):
            k = k.to(torch.get_default_dtype())
        self.k = torch.nn.Parameter(k, requires_grad=learnable)

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        dn = x.size(dim) - 1
        x = x**2
        quad_form = -x.narrow(dim, 0, 1) + x.narrow(dim, 1, dn).sum(
            dim=dim, keepdim=True
        )
        ok = torch.allclose(quad_form, -self.k, atol=atol, rtol=rtol)
        if not ok:
            reason = f"'x' minkowski quadratic form is not equal to {-self.k.item()}"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        inner_ = math.inner(u, x, dim=dim)
        ok = torch.allclose(inner_, torch.zeros(1, device=inner_.device), atol=atol, rtol=rtol)
        if not ok:
            reason = "Minkowski inner product is not equal to zero"
        else:
            reason = None
        return ok, reason

    def dist(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim)

    @__scaling__(ScalingInfo(1))
    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        return math.dist0(x, k=self.k, dim=dim, keepdim=keepdim)

    def norm(self, u: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return math.norm(u, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.egrad2rgrad(x, u, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.project(x, k=self.k, dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        v = math.project_u(x, v, k=self.k, dim=dim)
        return v

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, norm_tan=True, project=True, dim=-1
    ) -> torch.Tensor:
        if norm_tan is True:
            u = self.proju(x, u, dim=dim)
        res = math.expmap(x, u, k=self.k, dim=dim)
        if project is True:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo(u=-1))
    def expmap0(self, u: torch.Tensor, *, project=True, dim=-1) -> torch.Tensor:
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap(x, y, k=self.k, dim=dim)

    @__scaling__(ScalingInfo(1))
    def logmap0(self, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0(y, k=self.k, dim=dim)

    def logmap0back(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0back(x, k=self.k, dim=dim)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1,
    ) -> torch.Tensor:
        # TODO: x argument for maintaining the support of optims
        if v is None:
            v = u
        return math.inner(u, v, dim=dim, keepdim=keepdim)

    def inner0(
        self,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1,
    ) -> torch.Tensor:
        return math.inner0(v, k=self.k, dim=dim, keepdim=keepdim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.egrad2rgrad(x, u, k=self.k, dim=dim)

    def transp(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0(y, u, k=self.k, dim=dim)

    def transp0back(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0back(x, u, k=self.k, dim=dim)

    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    @__scaling__(ScalingInfo(t=-1))
    def geodesic_unit(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.geodesic_unit(t, x, u, k=self.k)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo(std=-1), "random")
    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        r"""
        Create a point on the manifold, measure is induced by Normal distribution on the tangent space of zero.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random points on Hyperboloid

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the projector `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the projector `dtype`, set the `dtype` arguement to None"
            )
        tens = torch.randn(*size, device=self.k.device, dtype=self.k.dtype) * std + mean
        tens /= tens.norm(dim=-1, keepdim=True)
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

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
            zero point on the manifold
        """
        if dtype is None:
            dtype = self.k.dtype
        if device is None:
            device = self.k.device

        zero_point = torch.zeros(*size, dtype=dtype, device=device)
        zero_point[..., 0] = torch.sqrt(self.k)
        return geoopt.ManifoldTensor(zero_point, manifold=self)

    retr = expmap
