import itertools
from typing import Tuple, Any, Union, List
import torch.jit
import functools
import operator
import lib.geoopt as geoopt

__all__ = [
    "strip_tuple",
    "size2shape",
    "make_tuple",
    "broadcast_shapes",
    "ismanifold",
    "canonical_manifold",
    "list_range",
    "idx2sign",
    "drop_dims",
    "canonical_dims",
    "sign",
    "prod",
    "clamp_abs",
    "sabs",
]

COMPLEX_DTYPES = {torch.complex64, torch.complex128}
if hasattr(torch, "complex32"):
    COMPLEX_DTYPES.add(torch.complex32)


def strip_tuple(tup: Tuple) -> Union[Tuple, Any]:
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def make_tuple(obj: Union[Tuple, List, Any]) -> Tuple:
    if isinstance(obj, list):
        obj = tuple(obj)
    if not isinstance(obj, tuple):
        return (obj,)
    else:
        return obj


def prod(items):
    return functools.reduce(operator.mul, items, 1)


@torch.jit.script
def sign(x):
    return torch.sign(x.sign() + 0.5)


@torch.jit.script
def sabs(x, eps: float = 1e-15):
    return x.abs().add_(eps)


@torch.jit.script
def clamp_abs(x, eps: float = 1e-15):
    s = sign(x)
    return s * sabs(x, eps=eps)


@torch.jit.script
def idx2sign(idx: int, dim: int, neg: bool = True):
    """
    Unify idx to be negative or positive, that helps in cases of broadcasting.

    Parameters
    ----------
    idx : int
        current index
    dim : int
        maximum dimension
    neg : bool
        indicate we need negative index

    Returns
    -------
    int
    """
    if neg:
        if idx < 0:
            return idx
        else:
            return (idx + 1) % -(dim + 1)
    else:
        return idx % dim


@torch.jit.script
def drop_dims(tensor: torch.Tensor, dims: List[int]):
    # Workaround to drop several dims in :func:`torch.squeeze`.
    seen: int = 0
    for d in dims:
        tensor = tensor.squeeze(d - seen)
        seen += 1
    return tensor


@torch.jit.script
def list_range(end: int):
    res: List[int] = []
    for d in range(end):
        res.append(d)
    return res


@torch.jit.script
def canonical_dims(dims: List[int], maxdim: int):
    result: List[int] = []
    for idx in dims:
        result.append(idx2sign(idx, maxdim, neg=False))
    return result


def size2shape(*size: Union[Tuple[int], int]) -> Tuple[int]:
    return make_tuple(strip_tuple(size))


def broadcast_shapes(*shapes: Tuple[int]) -> Tuple[int]:
    """Apply numpy broadcasting rules to shapes."""
    result = []
    for dims in itertools.zip_longest(*map(reversed, shapes), fillvalue=1):
        dim: int = 1
        for d in dims:
            if dim != 1 and d != 1 and d != dim:
                raise ValueError("Shapes can't be broadcasted")
            elif d > dim:
                dim = d
        result.append(dim)
    return tuple(reversed(result))


def ismanifold(instance, cls):
    """
    Check if interface of an instance is compatible with given class.

    Parameters
    ----------
    instance : geoopt.Manifold
        check if a given manifold is compatible with cls API
    cls : type
        manifold type

    Returns
    -------
    bool
        comparison result
    """
    if not issubclass(cls, geoopt.manifolds.Manifold):
        raise TypeError("`cls` should be a subclass of geoopt.manifolds.Manifold")
    if not isinstance(instance, geoopt.manifolds.Manifold):
        return False
    else:
        # this is the case to care about, Scaled class is a proxy, but fails instance checks
        while isinstance(instance, geoopt.Scaled):
            instance = instance.base
        return isinstance(instance, cls)


def canonical_manifold(manifold: "geoopt.Manifold"):
    """
    Get a canonical manifold.

    If a manifold is wrapped with Scaled. Some attributes may not be available. This should help if you really need them.

    Parameters
    ----------
    manifold : geoopt.Manifold

    Returns
    -------
    geoopt.Maniflold
        an unwrapped manifold
    """
    while isinstance(manifold, geoopt.Scaled):
        manifold = manifold.base
    return manifold
