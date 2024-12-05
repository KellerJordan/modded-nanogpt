from typing import Optional, Tuple, Union
import torch
from lib.geoopt import linalg as lalg
from lib.geoopt.utils import COMPLEX_DTYPES
from .siegel import SiegelManifold
from .upper_half import UpperHalf
from .vvd_metrics import SiegelMetricType
from ..siegel import csym_math as sm

__all__ = ["BoundedDomain"]


class BoundedDomain(SiegelManifold):
    r"""
    Bounded domain Manifold.

    This model generalizes the Poincare ball model.
    Points in the space are complex symmetric matrices.

    .. math::

        \mathcal{B}_n := \{ Z \in \operatorname{Sym}(n, \mathbb{C}) | Id - Z^*Z >> 0 \}

    Parameters
    ----------
    metric: SiegelMetricType
        one of Riemannian, Finsler One, Finsler Infinity, Finsler metric of minimum entropy, or learnable weighted sum.
    rank: int
        Rank of the space. Only mandatory for "fmin" and "wsum" metrics.
    """

    name = "Bounded Domain"

    def __init__(
        self, metric: SiegelMetricType = SiegelMetricType.RIEMANNIAN, rank: int = None
    ):
        super().__init__(metric=metric, rank=rank)

    def dist(
        self, z1: torch.Tensor, z2: torch.Tensor, *, keepdim=False
    ) -> torch.Tensor:
        """
        Compute distance in the Bounded domain model.

        To compute distances in the Bounded Domain Model we need to map the elements to the
        Upper Half Space Model by means of the Cayley Transform, and then compute distances
        in that domain.

        Parameters
        ----------
        z1 : torch.Tensor
             point on the manifold
        z2 : torch.Tensor
             point on the manifold
        keepdim : bool, optional
            keep the last dim?, by default False

        Returns
        -------
        torch.Tensor
            distance between two points
        """
        uhsm_z1 = sm.cayley_transform(z1)
        uhsm_z2 = sm.cayley_transform(z2)
        return super().dist(uhsm_z1, uhsm_z2)

    def egrad2rgrad(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`Z`.

        For a function :math:`f(Z)` on :math:`\mathcal{B}_n`, the gradient is:

        .. math::

            \operatorname{grad}_{R}(f(Z)) = A \cdot \operatorname{grad}_E(f(Z)) \cdot A

        where :math:`A = Id - \overline{Z}Z`

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
        a = get_id_minus_conjugate_z_times_z(z)
        return lalg.sym(a @ u @ a)  # impose symmetry due to numerical instabilities

    def projx(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Project point :math:`Z` on the manifold.

        In the Bounded domain model, we need to ensure that
        :math:`Id - \overline(Z)Z` is positive definite.

        Steps to project: Z complex symmetric matrix
        1) Diagonalize Z: :math:`Z = \overline{S} D S^*`
        2) Clamp eigenvalues: :math:`D' = clamp(D, max=1 - epsilon)`
        3) Rebuild Z: :math:`Z' = \overline{S} D' S^*`

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

        evalues, s = sm.takagi_eig(z)
        eps = sm.EPS[evalues.dtype]
        evalues_tilde = torch.clamp(evalues, max=1 - eps)

        diag_tilde = torch.diag_embed(evalues_tilde).type_as(z)
        z_tilde = s.conj() @ diag_tilde @ s.conj().transpose(-1, -2)

        # we do this so no operation is applied on the points that already belong to the space.
        # This prevents modifying values due to numerical instabilities
        batch_wise_mask = torch.all(evalues < 1 - eps, dim=-1, keepdim=True)
        already_in_space_mask = batch_wise_mask.unsqueeze(-1).expand_as(z)
        return torch.where(already_in_space_mask, z, z_tilde)

    def inner(
        self, z: torch.Tensor, u: torch.Tensor, v=None, *, keepdim=False
    ) -> torch.Tensor:
        r"""
        Inner product for tangent vectors at point :math:`Z`.

        The inner product at point :math:`Z = X + iY` of the vectors :math:`U, V` is:

        .. math::

            g_{Z}(U, V) = \operatorname{Tr}[(Id - \overline{Z}Z)^{-1} U (Id - Z\overline{Z})^{-1} \overline{V}]

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
        identity = sm.identity_like(z)
        conj_z = z.conj()

        inv_id_minus_conjz_z = sm.inverse(identity - (conj_z @ z))
        inv_id_minus_z_conjz = sm.inverse(identity - (z @ conj_z))

        res = inv_id_minus_conjz_z @ u @ inv_id_minus_z_conjz @ v.conj()
        return lalg.trace(res, keepdim=keepdim)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-4, rtol=1e-5):
        if not self._check_matrices_are_symmetric(x, atol=atol, rtol=rtol):
            return False, "Matrices are not symmetric"

        # Id - \overline{Z}Z is Hermitian and should be positive definite
        id_minus_zz = get_id_minus_conjugate_z_times_z(x)
        ok = torch.all(sm.eigvalsh(id_minus_zz) > 0)
        reason = None if ok else "'Id - overline{Z}Z' is not definite positive"
        return ok, reason

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        points = UpperHalf().random(*size, dtype=dtype, device=device, **kwargs)
        return sm.inverse_cayley_transform(points)

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
        seed: Optional[int] = 42,
    ) -> torch.Tensor:
        """
        Create points at the origin of the manifold in a deterministic way.

        For the Bounded domain model, the origin is the zero matrix.
        This is, a matrix whose real and imaginary parts are all zeros.

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
        if dtype and dtype not in COMPLEX_DTYPES:
            raise ValueError(f"dtype must be one of {COMPLEX_DTYPES}")
        if dtype is None:
            dtype = torch.complex128
        return torch.zeros(*size, dtype=dtype, device=device)


def get_id_minus_conjugate_z_times_z(z: torch.Tensor):
    r"""Given a complex symmetric matrix :math:`Z`, it returns an Hermitian matrix :math:`Id - \overline{Z}Z`."""
    return sm.identity_like(z) - (z.conj() @ z)
