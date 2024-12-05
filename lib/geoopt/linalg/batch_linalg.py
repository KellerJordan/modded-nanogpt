from typing import List, Callable, Tuple
import torch
import torch.jit
from functools import lru_cache, partial

__all__ = [
    "svd",
    "qr",
    "sym",
    "extract_diag",
    "matrix_rank",
    "expm",
    "trace",
    "block_matrix",
    "sym_funcm",
    "sym_expm",
    "sym_logm",
    "sym_sqrtm",
    "sym_invm",
    "sym_inv_sqrtm1",
    "sym_inv_sqrtm2",
]


@torch.jit.script
def sym(x: torch.Tensor):  # pragma: no cover
    return 0.5 * (x.transpose(-1, -2) + x)


@torch.jit.script
def extract_diag(x: torch.Tensor):  # pragma: no cover
    return torch.diagonal(x, 0, -1, -2)


matrix_rank = torch.linalg.matrix_rank

expm = torch.matrix_exp


@torch.jit.script
def block_matrix(blocks: List[List[torch.Tensor]], dim0: int = -2, dim1: int = -1):
    # [[A, B], [C, D]] ->
    # [AB]
    # [CD]
    hblocks = []
    for mats in blocks:
        hblocks.append(torch.cat(mats, dim=dim1))
    return torch.cat(hblocks, dim=dim0)


@torch.jit.script
def trace(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    r"""self-implemented matrix trace, since `torch.trace` only support 2-d input.

    Parameters
    ----------
    x : torch.Tensor
        input matrix
    keepdim : bool
            keep the last dim?

    Returns
    -------
    torch.Tensor
        :math:`\operatorname{Tr}(x)`
    """
    return torch.diagonal(x, dim1=-2, dim2=-1).sum(-1, keepdim=keepdim)


@lru_cache(None)
def _sym_funcm_impl(func, **kwargs):
    func = partial(func, **kwargs)

    def _impl(x):
        e, v = torch.linalg.eigh(x, "U")
        return v @ torch.diag_embed(func(e)) @ v.transpose(-1, -2)

    return torch.jit.script(_impl)


def sym_funcm(
    x: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Apply function to symmetric matrix.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    func : Callable[[torch.Tensor], torch.Tensor]
        function to apply

    Returns
    -------
    torch.Tensor
        symmetric matrix with function applied to
    """
    return _sym_funcm_impl(func)(x)


def sym_expm(x: torch.Tensor) -> torch.Tensor:
    r"""Symmetric matrix exponent.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    torch.Tensor
        :math:`\exp(x)`

    Notes
    -----
    Naive implementation of `torch.matrix_exp` seems to be fast enough
    """
    return expm(x)


def sym_logm(x: torch.Tensor) -> torch.Tensor:
    r"""Symmetric matrix logarithm.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    torch.Tensor
        :math:`\log(x)`
    """
    return sym_funcm(x, torch.log)


def sym_sqrtm(x: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix square root.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    torch.Tensor
        :math:`x^{1/2}`
    """
    return sym_funcm(x, torch.sqrt)


def sym_invm(x: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix inverse.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    torch.Tensor
        :math:`x^{-1}`

    Notes
    -----
    Naive implementation using `torch.matrix_power` seems to be fast enough
    """
    return torch.matrix_power(x, -1)


def sym_inv_sqrtm1(x: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix inverse square root.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    torch.Tensor
        :math:`x^{-1/2}`
    """
    return _sym_funcm_impl(torch.pow, exponent=-0.5)(x)


@torch.jit.script
def sym_inv_sqrtm2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric matrix inverse square root, with square root return also.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        :math:`x^{-1/2}`, :math:`x^{1/2}`
    """
    e, v = torch.linalg.eigh(x, "U")
    sqrt_e = torch.sqrt(e)
    inv_sqrt_e = torch.reciprocal(sqrt_e)
    return (
        v @ torch.diag_embed(inv_sqrt_e) @ v.transpose(-1, -2),
        v @ torch.diag_embed(sqrt_e) @ v.transpose(-1, -2),
    )


# left here for convenience
qr = torch.linalg.qr

svd = torch.linalg.svd
