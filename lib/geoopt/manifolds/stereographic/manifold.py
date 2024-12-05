import torch.nn
from typing import Tuple, Optional, List
from . import math
import lib.geoopt as geoopt
from ...utils import size2shape, broadcast_shapes
from ..base import Manifold, ScalingInfo

__all__ = [
    "Stereographic",
    "StereographicExact",
    "PoincareBall",
    "PoincareBallExact",
    "SphereProjection",
    "SphereProjectionExact",
]

_stereographic_doc = r"""
    :math:`\kappa`-Stereographic model.

    Parameters
    ----------
    k : float|tensor
        sectional curvature :math:`\kappa` of the manifold
        - k<0: Poincaré ball (stereographic projection of hyperboloid)
        - k>0: Stereographic projection of sphere
        - k=0: Euclidean geometry

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision.

    Documentation & Illustration
    ----------------------------
    http://andbloch.github.io/K-Stereographic-Model/ or :doc:`/extended/stereographic`
"""

_references = """References
    ----------
    The functions for the mathematics in gyrovector spaces are taken from the
    following resources:

    [1] Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic
           neural networks." Advances in neural information processing systems.
           2018.
    [2] Bachmann, Gregor, Gary Bécigneul, and Octavian-Eugen Ganea. "Constant
           Curvature Graph Convolutional Networks." arXiv preprint
           arXiv:1911.05076 (2019).
    [3] Skopek, Ondrej, Octavian-Eugen Ganea, and Gary Bécigneul.
           "Mixed-curvature Variational Autoencoders." arXiv preprint
           arXiv:1911.08411 (2019).
    [4] Ungar, Abraham A. Analytic hyperbolic geometry: Mathematical
           foundations and applications. World Scientific, 2005.
    [5] Albert, Ungar Abraham. Barycentric calculus in Euclidean and
           hyperbolic geometry: A comparative introduction. World Scientific,
           2010.
"""

_poincare_ball_doc = r"""
    Poincare ball model.

    See more in :doc:`/extended/stereographic`

    Parameters
    ----------
    c : float|tensor
        ball's negative curvature. The parametrization is constrained to have positive c

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision
"""

_sphere_projection_doc = r"""
    Stereographic Projection Spherical model.

    See more in :doc:`/extended/stereographic`

    Parameters
    ----------
    k : float|tensor
        sphere's positive curvature. The parametrization is constrained to have positive k

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision
"""


# noinspection PyMethodOverriding
class Stereographic(Manifold):
    __doc__ = r"""{}

    {}

    See Also
    --------
    :class:`StereographicExact`
    :class:`PoincareBall`
    :class:`PoincareBallExact`
    :class:`SphereProjection`
    :class:`SphereProjectionExact`
    """.format(
        _stereographic_doc,
        _references,
    )

    ndim = 1
    reversible = False
    name = property(lambda self: self.__class__.__name__)
    __scaling__ = Manifold.__scaling__.copy()

    @property
    def radius(self):
        return self.k.abs().sqrt().reciprocal()

    def __init__(self, k=0.0, learnable=False):
        super().__init__()
        k = torch.as_tensor(k)
        if not torch.is_floating_point(k):
            k = k.to(torch.get_default_dtype())
        self.k = torch.nn.Parameter(k, requires_grad=learnable)

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        px = math.project(x, k=self.k, dim=dim)
        ok = torch.allclose(x, px, atol=atol, rtol=rtol)
        if not ok:
            reason = "'x' norm lies out of the bounds [-1/sqrt(c)+eps, 1/sqrt(c)-eps]"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        return True, None

    def dist(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim)

    def dist2(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim) ** 2

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.egrad2rgrad(x, u, k=self.k, dim=dim)

    def retr(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        # always assume u is scaled properly
        approx = x + u
        return math.project(approx, k=self.k, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.project(x, k=self.k, dim=dim)

    def proju(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1,
    ) -> torch.Tensor:
        if v is None:
            v = u
        return math.inner(x, u, v, k=self.k, keepdim=keepdim, dim=dim)

    def norm(
        self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return math.norm(x, u, k=self.k, keepdim=keepdim, dim=dim)

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, project=True, dim=-1
    ) -> torch.Tensor:
        res = math.expmap(x, u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap(x, y, k=self.k, dim=dim)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1):
        return math.parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        y = self.retr(x, u, dim=dim)
        return self.transp(x, y, v, dim=dim)

    def transp_follow_expmap(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        *,
        dim=-1,
        project=True,
    ) -> torch.Tensor:
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    def expmap_transp(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        *,
        dim=-1,
        project=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.expmap(x, u, dim=dim, project=project)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.retr(x, u, dim=dim)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def mobius_add(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_add(x, y, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_sub(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_sub(x, y, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_coadd(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_coadd(x, y, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_cosub(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_cosub(x, y, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_scalar_mul(
        self, r: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_scalar_mul(r, x, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_pointwise_mul(
        self, w: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_pointwise_mul(w, x, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_matvec(
        self, m: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_matvec(m, x, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def geodesic(
        self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.geodesic(t, x, y, k=self.k, dim=dim)

    @__scaling__(ScalingInfo(t=-1))
    def geodesic_unit(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        dim=-1,
        project=True,
    ) -> torch.Tensor:
        res = math.geodesic_unit(t, x, u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def lambda_x(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        return math.lambda_x(x, k=self.k, dim=dim, keepdim=keepdim)

    @__scaling__(ScalingInfo(1))
    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        return math.dist0(x, k=self.k, dim=dim, keepdim=keepdim)

    @__scaling__(ScalingInfo(u=-1))
    def expmap0(self, u: torch.Tensor, *, dim=-1, project=True) -> torch.Tensor:
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo(1))
    def logmap0(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0(x, k=self.k, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0(y, u, k=self.k, dim=dim)

    def transp0back(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0back(y, u, k=self.k, dim=dim)

    def gyration(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.gyration(x, y, z, k=self.k, dim=dim)

    def antipode(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.antipode(x, k=self.k, dim=dim)

    @__scaling__(ScalingInfo(1))
    def dist2plane(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        a: torch.Tensor,
        *,
        dim=-1,
        keepdim=False,
        signed=False,
        scaled=False,
    ) -> torch.Tensor:
        return math.dist2plane(
            x,
            p,
            a,
            dim=dim,
            k=self.k,
            keepdim=keepdim,
            signed=signed,
            scaled=scaled,
        )

    # this does not yet work with scaling
    @__scaling__(ScalingInfo.NotCompatible)
    def mobius_fn_apply(
        self,
        fn: callable,
        x: torch.Tensor,
        *args,
        dim=-1,
        project=True,
        **kwargs,
    ) -> torch.Tensor:
        res = math.mobius_fn_apply(fn, x, *args, k=self.k, dim=dim, **kwargs)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    # this does not yet work with scaling
    @__scaling__(ScalingInfo.NotCompatible)
    def mobius_fn_apply_chain(
        self, x: torch.Tensor, *fns: callable, project=True, dim=-1
    ) -> torch.Tensor:
        res = math.mobius_fn_apply_chain(x, *fns, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo(std=-1), "random")
    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        """
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
            random point on the PoincareBall manifold

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        size = size2shape(*size)
        self._assert_check_shape(size, "x")
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the manifold `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the manifold `dtype`, set the `dtype` argument to None"
            )
        tens = (
            torch.randn(size, device=self.k.device, dtype=self.k.dtype)
            * std
            / size[-1] ** 0.5
            + mean
        )
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    random = random_normal

    @__scaling__(ScalingInfo(std=-1))
    def wrapped_normal(
        self, *size, mean: torch.Tensor, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        """
        Create a point on the manifold, measure is induced by Normal distribution on the tangent space of mean.

        Definition is taken from
        [1] Mathieu, Emile et. al. "Continuous Hierarchical Representations with
        Poincaré Variational Auto-Encoders." arXiv preprint
        arxiv:1901.06033 (2019).

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
            random point on the PoincareBall manifold

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        size = size2shape(*size)
        self._assert_check_shape(size, "x")
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the manifold `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the manifold `dtype`, set the `dtype` argument to None"
            )
        v = torch.randn(size, device=self.k.device, dtype=self.k.dtype) * std
        lambda_x = self.lambda_x(mean).unsqueeze(-1)
        return geoopt.ManifoldTensor(self.expmap(mean, v / lambda_x), manifold=self)

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
            random point on the manifold
        """
        return geoopt.ManifoldTensor(
            torch.zeros(*size, dtype=dtype, device=device), manifold=self
        )

    def weighted_midpoint(
        self,
        xs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        *,
        reducedim: Optional[List[int]] = None,
        dim: int = -1,
        keepdim: bool = False,
        lincomb: bool = False,
        posweight=False,
        project=True,
    ):
        mid = math.weighted_midpoint(
            xs=xs,
            weights=weights,
            k=self.k,
            reducedim=reducedim,
            dim=dim,
            keepdim=keepdim,
            lincomb=lincomb,
            posweight=posweight,
        )
        if project:
            return math.project(mid, k=self.k, dim=dim)
        else:
            return mid

    def sproj(self, x: torch.Tensor, *, dim: int = -1):
        return math.sproj(x, k=self.k, dim=dim)

    def inv_sproj(self, x: torch.Tensor, *, dim: int = -1):
        return math.inv_sproj(x, k=self.k, dim=dim)


class StereographicExact(Stereographic):
    __doc__ = r"""{}

    The implementation of retraction is an exact exponential map, this retraction will be used in optimization.

    See Also
    --------
    :class:`Stereographic`
    :class:`PoincareBall`
    :class:`PoincareBallExact`
    :class:`SphereProjection`
    :class:`SphereProjectionExact`
    """.format(
        _stereographic_doc
    )

    reversible = True
    retr_transp = Stereographic.expmap_transp
    transp_follow_retr = Stereographic.transp_follow_expmap
    retr = Stereographic.expmap

    def extra_repr(self):
        return "exact"


class PoincareBall(Stereographic):
    __doc__ = r"""{}

    See Also
    --------
    :class:`Stereographic`
    :class:`StereographicExact`
    :class:`PoincareBallExact`
    :class:`SphereProjection`
    :class:`SphereProjectionExact`
    """.format(
        _poincare_ball_doc
    )

    @property
    def k(self):
        return -self.c

    @property
    def c(self):
        return torch.nn.functional.softplus(self.isp_c)

    def __init__(self, c=1.0, learnable=False):
        super().__init__(k=c, learnable=learnable)
        k = self._parameters.pop("k")
        with torch.no_grad():
            self.isp_c = k.exp_().sub_(1).log_()


class PoincareBallExact(PoincareBall, StereographicExact):
    __doc__ = r"""{}

    The implementation of retraction is an exact exponential map, this retraction will be used in optimization.

    See Also
    --------
    :class:`Stereographic`
    :class:`StereographicExact`
    :class:`PoincareBall`
    :class:`SphereProjection`
    :class:`SphereProjectionExact`
    """.format(
        _poincare_ball_doc
    )


class SphereProjection(Stereographic):
    __doc__ = r"""{}

    See Also
    --------
    :class:`Stereographic`
    :class:`StereographicExact`
    :class:`PoincareBall`
    :class:`PoincareBallExact`
    :class:`SphereProjectionExact`
    :class:`Sphere`
    """.format(
        _sphere_projection_doc
    )

    @property
    def k(self):
        return torch.nn.functional.softplus(self.isp_k)

    def __init__(self, k=1.0, learnable=False):
        super().__init__(k=k, learnable=learnable)
        k = self._parameters.pop("k")
        with torch.no_grad():
            self.isp_k = k.exp_().sub_(1).log_()


class SphereProjectionExact(SphereProjection, StereographicExact):
    __doc__ = r"""{}

    The implementation of retraction is an exact exponential map, this retraction will be used in optimization.

    See Also
    --------
    :class:`Stereographic`
    :class:`StereographicExact`
    :class:`PoincareBall`
    :class:`PoincareBallExact`
    :class:`SphereProjectionExact`
    :class:`Sphere`
    """.format(
        _sphere_projection_doc
    )
