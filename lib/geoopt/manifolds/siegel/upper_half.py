from typing import Optional, Tuple, Union
import torch
from lib.geoopt import linalg as lalg
from lib.geoopt.utils import COMPLEX_DTYPES
from .siegel import SiegelManifold
from .vvd_metrics import SiegelMetricType
from ..siegel import csym_math as sm

__all__ = ["UpperHalf"]


class UpperHalf(SiegelManifold):
    r"""
    Upper Half Space Manifold.

    This model generalizes the upper half plane model of the hyperbolic plane.
    Points in the space are complex symmetric matrices.

    .. math::

        \mathcal{S}_n = \{Z = X + iY \in \operatorname{Sym}(n, \mathbb{C}) | Y >> 0 \}.


    Parameters
    ----------
    metric: SiegelMetricType
        one of Riemannian, Finsler One, Finsler Infinity, Finsler metric of minimum entropy, or learnable weighted sum.
    rank: int
        Rank of the space. Only mandatory for "fmin" and "wsum" metrics.
    """

    name = "Upper Half Space"

    def __init__(
        self, metric: SiegelMetricType = SiegelMetricType.RIEMANNIAN, rank: int = None
    ):
        super().__init__(metric=metric, rank=rank)

    def egrad2rgrad(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`Z`.

        For a function :math:`f(Z)` on :math:`\mathcal{S}_n`, the gradient is:

        .. math::

            \operatorname{grad}_{R}(f(Z)) = Y \cdot \operatorname{grad}_E(f(Z)) \cdot Y

        where :math:`Y` is the imaginary part of :math:`Z`.

        Parameters
        ----------
        z : torch.Tensor
             point on the manifold
        u : torch.Tensor
             gradient to be projected

        Returns
        -------
        torch.Tensor
            Riemannian gradient
        """
        real_grad, imag_grad = u.real, u.imag
        y = z.imag
        real_grad = y @ real_grad @ y
        imag_grad = y @ imag_grad @ y
        return lalg.sym(
            sm.to_complex(real_grad, imag_grad)
        )  # impose symmetry due to numerical instabilities

    def projx(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project point :math:`Z` on the manifold.

        In this space, we need to ensure that :math:`Y = Im(Z)` is positive definite.
        Since the matrix Y is symmetric, it is possible to diagonalize it.
        For a diagonal matrix the condition is just that all diagonal entries are positive,
        so we clamp the values that are <= 0 in the diagonal to an epsilon, and then restore
        the matrix back into non-diagonal form using the base change matrix that was obtained
        from the diagonalization.

        Parameters
        ----------
        z : torch.Tensor
             point on the manifold

        Returns
        -------
        torch.Tensor
            Projected points
        """
        z = super().projx(z)
        y = sm.positive_conjugate_projection(z.imag)
        return sm.to_complex(z.real, y)

    def inner(
        self, z: torch.Tensor, u: torch.Tensor, v=None, *, keepdim=False
    ) -> torch.Tensor:
        r"""
        Inner product for tangent vectors at point :math:`Z`.

        The inner product at point :math:`Z = X + iY` of the vectors :math:`U, V` is:

        .. math::

            g_{Z}(U, V) = \operatorname{Tr}[ Y^{-1} U Y^{-1} \overline{V} ]

        Parameters
        ----------
        z : torch.Tensor
             point on the manifold
        u : torch.Tensor
             tangent vector at point :math:`z`
        v : torch.Tensor
             tangent vector at point :math:`z`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            inner product (broadcasted)
        """
        if v is None:
            v = u
        inv_y = sm.inverse(z.imag).type_as(z)

        res = inv_y @ u @ inv_y @ v.conj()
        return lalg.trace(res, keepdim=keepdim)

    def _check_point_on_manifold(self, z: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        if not self._check_matrices_are_symmetric(z, atol=atol, rtol=rtol):
            return False, "Matrices are not symmetric"

        # Im(Z) should be positive definite.
        ok = torch.all(sm.eigvalsh(z.imag) > 0)
        if not ok:
            reason = "Imaginary part of Z is not positive definite"
        else:
            reason = None
        return ok, reason

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        if dtype and dtype not in COMPLEX_DTYPES:
            raise ValueError(f"dtype must be one of {COMPLEX_DTYPES}")
        if dtype is None:
            dtype = torch.complex128
        tens = 0.5 * torch.randn(*size, dtype=dtype, device=device)
        tens = lalg.sym(tens)
        tens.imag = lalg.expm(tens.imag)
        return tens

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
        seed: Optional[int] = 42,
    ) -> torch.Tensor:
        """
        Create points at the origin of the manifold in a deterministic way.

        For the Upper half model, the origin is the imaginary identity.
        This is, a matrix whose real part is all zeros, and the identity as the imaginary part.

        Parameters
        ----------
        size : Union[int, Tuple[int]]
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : Optional[int]
            A parameter controlling deterministic randomness for manifolds that do not provide ``.origin``,
            but provide ``.random``. (default: 42)

        Returns
        -------
        torch.Tensor
        """
        imag = torch.eye(*size[:-1], dtype=dtype, device=device)
        if imag.dtype in COMPLEX_DTYPES:
            imag = imag.real
        return torch.complex(torch.zeros_like(imag), imag)
