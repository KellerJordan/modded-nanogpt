from abc import ABC
from typing import Union, Tuple, Optional
import torch
from ..base import Manifold
from lib.geoopt import linalg as lalg
from ..siegel import csym_math as sm
from .vvd_metrics import SiegelMetricType, SiegelMetricFactory


class SiegelManifold(Manifold, ABC):
    """Abstract Manifold to work on Siegel spaces.

    The implementation is aimed to work with realization of the Siegel space as
    spaces of complex symmetric matrices.

    References
    ----------
    - Federico López, Beatrice Pozzetti, Steve Trettel, Michael Strube, Anna Wienhard.
      "Symmetric Spaces for Graph Embeddings: A Finsler-Riemannian Approach", 2021.

    Parameters
    ----------
    metric: SiegelMetricType
        one of Riemannian, Finsler One, Finsler Infinity, Finsler metric of minimum entropy, or learnable weighted sum.
    rank: int
        Rank of the space. Only mandatory for "fmin" and "wsum" metrics.
    """

    __scaling__ = Manifold.__scaling__.copy()
    name = "Siegel Space"
    ndim = 2
    reversible = False

    def __init__(
        self, metric: SiegelMetricType = SiegelMetricType.RIEMANNIAN, rank: int = None
    ):
        super().__init__()
        self.metric = SiegelMetricFactory.get(metric, rank)

    def dist(
        self, z1: torch.Tensor, z2: torch.Tensor, *, keepdim=False
    ) -> torch.Tensor:
        """
        Compute distance between two points on the manifold according to the specified metric.

        Calculates the distance for the Upper Half Space Manifold (UHSM)
        It is implemented here since the way to calculate distances in the Bounded Domain Manifold
        requires mapping the points to the UHSM, and then applying this formula.

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
        # with Z1 = X + iY, define Z3 = sqrt(Y)^-1 (Z2 - X) sqrt(Y)^-1
        x, y = z1.real, z1.imag
        inv_sqrt_y = lalg.sym_inv_sqrtm1(y).type_as(z1)
        z3 = inv_sqrt_y @ (z2 - x) @ inv_sqrt_y

        w = sm.inverse_cayley_transform(z3)
        evalues = sm.takagi_eigvals(w)  # evalues are in ascending order e1 < e2 < en

        # assert 0 <= evalues <= 1
        eps = sm.EPS[evalues.dtype]
        assert torch.all(evalues >= 0 - eps), f"Eigenvalues: {evalues}"
        assert torch.all(evalues <= 1.01), f"Eigenvalues: {evalues}"

        # Vector-valued distance: v_i = log((1 + e_i) / (1 - e_i))
        vvd = (1 + evalues) / (1 - evalues).clamp(min=eps)
        vvd = torch.log(vvd)
        res = self.metric.compute_metric(vvd)
        return res

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # always assume u is scaled properly
        approx = x + u
        return self.projx(approx)

    def _check_matrices_are_symmetric(
        self, x: torch.Tensor, *, atol: float = 1e-4, rtol: float = 1e-5
    ):
        """Check that matrices are symmetric.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        atol : float
            absolute tolerance for allclose
        rtol : float
            relative tolerance for allclose

        Returns
        -------
        boolean
            whether the points in x are complex symmetric or not
        """
        return sm.is_complex_symmetric(x, atol, rtol)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        return lalg.sym(x)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.egrad2rgrad(x, u)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(u, u.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return (
                False,
                "u is not symmetric (u != u.transpose) with atol={}, rtol={}".format(
                    atol, rtol
                ),
            )
        return True, None

    def extra_repr(self) -> str:
        return f"metric={type(self.metric).__name__}"
