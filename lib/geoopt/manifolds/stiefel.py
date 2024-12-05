import torch
from typing import Union, Tuple, Optional
from .base import Manifold
from .. import linalg
from ..utils import size2shape
from ..tensor import ManifoldTensor


__all__ = ["Stiefel", "EuclideanStiefel", "CanonicalStiefel", "EuclideanStiefelExact"]


_stiefel_doc = r"""
    Manifold induced by the following matrix constraint:

    .. math::

        X^\top X = I\\
        X \in \mathrm{R}^{n\times m}\\
        n \ge m
"""


class Stiefel(Manifold):
    __doc__ = r"""
    {}

    Parameters
    ----------
    canonical : bool
        Use canonical inner product instead of euclidean one (defaults to canonical)

    See Also
    --------
    :class:`CanonicalStiefel`, :class:`EuclideanStiefel`, :class:`EuclideanStiefelExact`
    """.format(
        _stiefel_doc
    )
    ndim = 2

    def __new__(cls, canonical=True):
        if cls is Stiefel:
            if canonical:
                return super().__new__(CanonicalStiefel)
            else:
                return super().__new__(EuclideanStiefel)
        else:
            return super().__new__(cls)

    def _check_shape(
        self, shape: Tuple[int], name: str
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok, reason = super()._check_shape(shape, name)
        if not ok:
            return False, reason
        shape_is_ok = shape[-1] <= shape[-2]
        if not shape_is_ok:
            return (
                False,
                "`{}` should have shape[-1] <= shape[-2], got {} </= {}".format(
                    name, shape[-1], shape[-2]
                ),
            )
        return True, None

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        xtx = x.transpose(-1, -2) @ x
        # less memory usage for substract diagonal
        xtx[..., torch.arange(x.shape[-1]), torch.arange(x.shape[-1])] -= 1
        ok = torch.allclose(xtx, xtx.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`X^T X != I` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        diff = u.transpose(-1, -2) @ x + x.transpose(-1, -2) @ u
        ok = torch.allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u^T x + x^T u !=0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        U, _, V = linalg.svd(x, full_matrices=False)
        return torch.einsum("...ik,...kj->...ij", U, V)

    def random_naive(self, *size, dtype=None, device=None) -> torch.Tensor:
        """
        Naive approach to get random matrix on Stiefel manifold.

        A helper function to sample a random point on the Stiefel manifold.
        The measure is non-uniform for this method, but fast to compute.

        Parameters
        ----------
        size : shape
            the desired output shape
        dtype : torch.dtype
            desired dtype
        device : torch.device
            desired device

        Returns
        -------
        ManifoldTensor
            random point on Stiefel manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        tens = torch.randn(*size, device=device, dtype=dtype)
        return ManifoldTensor(linalg.qr(tens)[0], manifold=self)

    random = random_naive

    def origin(self, *size, dtype=None, device=None, seed=42) -> torch.Tensor:
        """
        Identity matrix point origin.

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
        eye = torch.zeros(*size, dtype=dtype, device=device)
        eye[..., torch.arange(eye.shape[-1]), torch.arange(eye.shape[-1])] += 1
        return ManifoldTensor(eye, manifold=self)


class CanonicalStiefel(Stiefel):
    __doc__ = r"""Stiefel Manifold with Canonical inner product

    {}
    """.format(
        _stiefel_doc
    )

    name = "Stiefel(canonical)"
    reversible = True

    @staticmethod
    def _amat(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u @ x.transpose(-1, -2) - x @ u.transpose(-1, -2)

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        # <u, v>_x = tr(u^T(I-1/2xx^T)v)
        # = tr(u^T(v-1/2xx^Tv))
        # = tr(u^Tv-1/2u^Txx^Tv)
        # = tr(u^Tv-1/2u^Txx^Tv)
        # = tr(u^Tv)-1/2tr(x^Tvu^Tx)
        # = \sum_ij{(u*v}_ij}-1/2\sum_ij{(x^Tv * x^Tu)_ij}
        xtu = x.transpose(-1, -2) @ u
        if v is None:
            xtv = xtu
            v = u
        else:
            xtv = x.transpose(-1, -2) @ v
        return (u * v).sum([-1, -2], keepdim=keepdim) - 0.5 * (xtv * xtu).sum(
            [-1, -2], keepdim=keepdim
        )

    def _transp_follow_one(
        self, x: torch.Tensor, v: torch.Tensor, *, u: torch.Tensor
    ) -> torch.Tensor:
        a = self._amat(x, u)
        rhs = v + 1 / 2 * a @ v
        lhs = -1 / 2 * a
        lhs[..., torch.arange(a.shape[-2]), torch.arange(x.shape[-2])] += 1
        qv = torch.linalg.solve(lhs, rhs)
        return qv

    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return self._transp_follow_one(x, v, u=u)

    transp_follow_expmap = transp_follow_retr

    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xvs = torch.cat((x, v), -1)
        qxvs = self._transp_follow_one(x, xvs, u=u).view(
            x.shape[:-1] + (2, x.shape[-1])
        )
        new_x, new_v = qxvs.unbind(-2)
        return new_x, new_v

    expmap_transp = retr_transp

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u - x @ u.transpose(-1, -2) @ x

    egrad2rgrad = proju

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self._transp_follow_one(x, x, u=u)

    expmap = retr


class EuclideanStiefel(Stiefel):
    __doc__ = r"""Stiefel Manifold with Euclidean inner product

    {}
    """.format(
        _stiefel_doc
    )

    name = "Stiefel(euclidean)"
    reversible = False

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u - x @ linalg.sym(x.transpose(-1, -2) @ u)

    egrad2rgrad = proju

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.proju(y, v)

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            v = u
        return (u * v).sum([-1, -2], keepdim=keepdim)

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q, r = linalg.qr(x + u)
        unflip = linalg.extract_diag(r).sign().add(0.5).sign()
        q *= unflip[..., None, :]
        return q

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xtu = x.transpose(-1, -2) @ u
        utu = u.transpose(-1, -2) @ u
        eye = torch.zeros_like(utu)
        eye[..., torch.arange(utu.shape[-2]), torch.arange(utu.shape[-2])] += 1
        logw = linalg.block_matrix(((xtu, -utu), (eye, xtu)))
        w = linalg.expm(logw)
        z = torch.cat((linalg.expm(-xtu), torch.zeros_like(utu)), dim=-2)
        y = torch.cat((x, u), dim=-1) @ w @ z
        return y


class EuclideanStiefelExact(EuclideanStiefel):
    __doc__ = r"""{}

    Notes
    -----
    The implementation of retraction is an exact exponential map, this retraction will be used in optimization
    """.format(
        EuclideanStiefel.__doc__
    )

    retr_transp = EuclideanStiefel.expmap_transp
    transp_follow_retr = EuclideanStiefel.transp_follow_expmap
    retr = EuclideanStiefel.expmap

    def extra_repr(self):
        return "exact"
