r"""
:math:`\kappa`-Stereographic math module.

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

import functools
import torch.jit
from typing import List, Optional
from ...utils import list_range, drop_dims, sign, clamp_abs, sabs


@torch.jit.script
def tanh(x):
    return x.clamp(-15, 15).tanh()


@torch.jit.script
def artanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)


@torch.jit.script
def arsinh(x: torch.Tensor):
    return (x + torch.sqrt(1 + x.pow(2))).clamp_min(1e-15).log().to(x.dtype)


@torch.jit.script
def abs_zero_grad(x):
    # this op has derivative equal to 1 at zero
    return x * sign(x)


@torch.jit.script
def tan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            + 1 / 3 * k * x**3
            + 2 / 15 * k**2 * x**5
            + 17 / 315 * k**3 * x**7
            + 62 / 2835 * k**4 * x**9
            + 1382 / 155925 * k**5 * x**11
            # + o(k**6)
        )
    elif order == 1:
        return x + 1 / 3 * k * x**3
    elif order == 2:
        return x + 1 / 3 * k * x**3 + 2 / 15 * k**2 * x**5
    elif order == 3:
        return (
            x
            + 1 / 3 * k * x**3
            + 2 / 15 * k**2 * x**5
            + 17 / 315 * k**3 * x**7
        )
    elif order == 4:
        return (
            x
            + 1 / 3 * k * x**3
            + 2 / 15 * k**2 * x**5
            + 17 / 315 * k**3 * x**7
            + 62 / 2835 * k**4 * x**9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")


@torch.jit.script
def artan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - 1 / 3 * k * x**3
            + 1 / 5 * k**2 * x**5
            - 1 / 7 * k**3 * x**7
            + 1 / 9 * k**4 * x**9
            - 1 / 11 * k**5 * x**11
            # + o(k**6)
        )
    elif order == 1:
        return x - 1 / 3 * k * x**3
    elif order == 2:
        return x - 1 / 3 * k * x**3 + 1 / 5 * k**2 * x**5
    elif order == 3:
        return (
            x - 1 / 3 * k * x**3 + 1 / 5 * k**2 * x**5 - 1 / 7 * k**3 * x**7
        )
    elif order == 4:
        return (
            x
            - 1 / 3 * k * x**3
            + 1 / 5 * k**2 * x**5
            - 1 / 7 * k**3 * x**7
            + 1 / 9 * k**4 * x**9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")


@torch.jit.script
def arsin_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            + k * x**3 / 6
            + 3 / 40 * k**2 * x**5
            + 5 / 112 * k**3 * x**7
            + 35 / 1152 * k**4 * x**9
            + 63 / 2816 * k**5 * x**11
            # + o(k**6)
        )
    elif order == 1:
        return x + k * x**3 / 6
    elif order == 2:
        return x + k * x**3 / 6 + 3 / 40 * k**2 * x**5
    elif order == 3:
        return x + k * x**3 / 6 + 3 / 40 * k**2 * x**5 + 5 / 112 * k**3 * x**7
    elif order == 4:
        return (
            x
            + k * x**3 / 6
            + 3 / 40 * k**2 * x**5
            + 5 / 112 * k**3 * x**7
            + 35 / 1152 * k**4 * x**9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")


@torch.jit.script
def sin_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - k * x**3 / 6
            + k**2 * x**5 / 120
            - k**3 * x**7 / 5040
            + k**4 * x**9 / 362880
            - k**5 * x**11 / 39916800
            # + o(k**6)
        )
    elif order == 1:
        return x - k * x**3 / 6
    elif order == 2:
        return x - k * x**3 / 6 + k**2 * x**5 / 120
    elif order == 3:
        return x - k * x**3 / 6 + k**2 * x**5 / 120 - k**3 * x**7 / 5040
    elif order == 4:
        return (
            x
            - k * x**3 / 6
            + k**2 * x**5 / 120
            - k**3 * x**7 / 5040
            + k**4 * x**9 / 362880
        )
    else:
        raise RuntimeError("order not in [-1, 5]")


@torch.jit.script
def tan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return tan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * tanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.clamp_max(1e38).tan()
    else:
        tan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.clamp_max(1e38).tan(), tanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, tan_k_zero_taylor(x, k, order=1), tan_k_nonzero)


@torch.jit.script
def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return artan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * artanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.atan()
    else:
        artan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.atan(), artanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, artan_k_zero_taylor(x, k, order=1), artan_k_nonzero)


@torch.jit.script
def arsin_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return arsin_k_zero_taylor(x, k)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * arsinh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.asin()
    else:
        arsin_k_nonzero = (
            torch.where(
                k_sign.gt(0),
                scaled_x.clamp(-1 + 1e-7, 1 - 1e-7).asin(),
                arsinh(scaled_x),
            )
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, arsin_k_zero_taylor(x, k, order=1), arsin_k_nonzero)


@torch.jit.script
def sin_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return sin_k_zero_taylor(x, k)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * torch.sinh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.sin()
    else:
        sin_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.sin(), torch.sinh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, sin_k_zero_taylor(x, k, order=1), sin_k_nonzero)


def project(x: torch.Tensor, *, k: torch.Tensor, dim=-1, eps=-1):
    r"""
    Safe projection on the manifold for numerical stability.

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension to compute norm
    eps : float
        stability parameter, uses default for dtype if not provided

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k, dim, eps)


@torch.jit.script
def _project(x, k, dim: int = -1, eps: float = -1.0):
    if eps < 0:
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
    maxnorm = (1 - eps) / (sabs(k) ** 0.5)
    maxnorm = torch.where(k.lt(0), maxnorm, k.new_full((), 1e15))
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def lambda_x(x: torch.Tensor, *, k: torch.Tensor, keepdim=False, dim=-1):
    r"""
    Compute the conformal factor :math:`\lambda^\kappa_x` for a point on the ball.

    .. math::
        \lambda^\kappa_x = \frac{2}{1 + \kappa \|x\|_2^2}

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        conformal factor
    """
    return _lambda_x(x, k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _lambda_x(x: torch.Tensor, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    return 2 / (1 + k * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(1e-15)


def inner(
    x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, k, keepdim=False, dim=-1
):
    r"""
    Compute inner product for two vectors on the tangent space w.r.t Riemannian metric on the Poincare ball.

    .. math::

        \langle u, v\rangle_x = (\lambda^\kappa_x)^2 \langle u, v \rangle

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    u : tensor
        tangent vector to :math:`x` on Poincare ball
    v : tensor
        tangent vector to :math:`x` on Poincare ball
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner(x, u, v, k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _inner(
    x: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    k: torch.Tensor,
    keepdim: bool = False,
    dim: int = -1,
):
    return _lambda_x(x, k, keepdim=True, dim=dim) ** 2 * (u * v).sum(
        dim=dim, keepdim=keepdim
    )


def norm(x: torch.Tensor, u: torch.Tensor, *, k: torch.Tensor, keepdim=False, dim=-1):
    r"""
    Compute vector norm on the tangent space w.r.t Riemannian metric on the Poincare ball.

    .. math::

        \|u\|_x = \lambda^\kappa_x \|u\|_2

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    u : tensor
        tangent vector to :math:`x` on Poincare ball
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        norm of vector
    """
    return _norm(x, u, k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _norm(
    x: torch.Tensor,
    u: torch.Tensor,
    k: torch.Tensor,
    keepdim: bool = False,
    dim: int = -1,
):
    return _lambda_x(x, k, keepdim=keepdim, dim=dim) * u.norm(
        dim=dim, keepdim=keepdim, p=2
    )


def mobius_add(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius gyrovector addition.

    .. math::

        x \oplus_\kappa y =
        \frac{
            (1 - 2 \kappa \langle x, y\rangle - \kappa \|y\|^2_2) x +
            (1 + \kappa \|x\|_2^2) y
        }{
            1 - 2 \kappa \langle x, y\rangle + \kappa^2 \|x\|^2_2 \|y\|^2_2
        }

    .. plot:: plots/extended/stereographic/mobius_add.py

    In general this operation is not commutative:

    .. math::

        x \oplus_\kappa y \ne y \oplus_\kappa x

    But in some cases this property holds:

    * zero vector case

    .. math::

        \mathbf{0} \oplus_\kappa x = x \oplus_\kappa \mathbf{0}

    * zero curvature case that is same as Euclidean addition

    .. math::

        x \oplus_0 y = y \oplus_0 x

    Another useful property is so called left-cancellation law:

    .. math::

        (-x) \oplus_\kappa (x \oplus_\kappa y) = y

    Parameters
    ----------
    x : tensor
        point on the manifold
    y : tensor
        point on the manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius addition
    """
    return _mobius_add(x, y, k, dim=dim)


@torch.jit.script
def _mobius_add(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    denom = 1 - 2 * k * xy + k**2 * x2 * y2
    # minimize denom (omit K to simplify th notation)
    # 1)
    # {d(denom)/d(x) = 2 y + 2x * <y, y> = 0
    # {d(denom)/d(y) = 2 x + 2y * <x, x> = 0
    # 2)
    # {y + x * <y, y> = 0
    # {x + y * <x, x> = 0
    # 3)
    # {- y/<y, y> = x
    # {- x/<x, x> = y
    # 4)
    # minimum = 1 - 2 <y, y>/<y, y> + <y, y>/<y, y> = 0
    return num / denom.clamp_min(1e-15)


def mobius_sub(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius gyrovector subtraction.

    The Möbius subtraction can be represented via the Möbius addition as
    follows:

    .. math::

        x \ominus_\kappa y = x \oplus_\kappa (-y)

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius subtraction
    """
    return _mobius_sub(x, y, k, dim=dim)


def _mobius_sub(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    return _mobius_add(x, -y, k, dim=dim)


def gyration(
    a: torch.Tensor, b: torch.Tensor, u: torch.Tensor, *, k: torch.Tensor, dim=-1
):
    r"""
    Compute the gyration of :math:`u` by :math:`[a,b]`.

    The gyration is a special operation of gyrovector spaces. The gyrovector
    space addition operation :math:`\oplus_\kappa` is not associative (as
    mentioned in :func:`mobius_add`), but it is gyroassociative, which means

    .. math::

        u \oplus_\kappa (v \oplus_\kappa w)
        =
        (u\oplus_\kappa v) \oplus_\kappa \operatorname{gyr}[u, v]w,

    where

    .. math::

        \operatorname{gyr}[u, v]w
        =
        \ominus (u \oplus_\kappa v) \oplus (u \oplus_\kappa (v \oplus_\kappa w))

    We can simplify this equation using the explicit formula for the Möbius
    addition [1]. Recall,

    .. math::

        A = - \kappa^2 \langle u, w\rangle \langle v, v\rangle
            - \kappa \langle v, w\rangle
            + 2 \kappa^2 \langle u, v\rangle \langle v, w\rangle\\
        B = - \kappa^2 \langle v, w\rangle \langle u, u\rangle
            + \kappa \langle u, w\rangle\\
        D = 1 - 2 \kappa \langle u, v\rangle
            + \kappa^2 \langle u, u\rangle \langle v, v\rangle\\

        \operatorname{gyr}[u, v]w = w + 2 \frac{A u + B v}{D}.

    Parameters
    ----------
    a : tensor
        first point on manifold
    b : tensor
        second point on manifold
    u : tensor
        vector field for operation
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of automorphism

    References
    ----------
    [1]  A. A. Ungar (2009), A Gyrovector Space Approach to Hyperbolic Geometry
    """
    return _gyration(a, b, u, k, dim=dim)


@torch.jit.script
def _gyration(
    u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    # non-simplified
    # mupv = -_mobius_add(u, v, K)
    # vpw = _mobius_add(u, w, K)
    # upvpw = _mobius_add(u, vpw, K)
    # return _mobius_add(mupv, upvpw, K)
    # simplified
    u2 = u.pow(2).sum(dim=dim, keepdim=True)
    v2 = v.pow(2).sum(dim=dim, keepdim=True)
    uv = (u * v).sum(dim=dim, keepdim=True)
    uw = (u * w).sum(dim=dim, keepdim=True)
    vw = (v * w).sum(dim=dim, keepdim=True)
    K2 = k**2
    a = -K2 * uw * v2 - k * vw + 2 * K2 * uv * vw
    b = -K2 * vw * u2 + k * uw
    d = 1 - 2 * k * uv + K2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(1e-15)


def mobius_coadd(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius gyrovector coaddition.

    The addition operation :math:`\oplus_\kappa` is neither associative, nor
    commutative. In contrast, the coaddition :math:`\boxplus_\kappa` (or
    cooperation) is an associative operation that is defined as follows.

    .. math::

        a \boxplus_\kappa b
        =
        b \boxplus_\kappa a
        =
        a\operatorname{gyr}[a, -b]b\\
        = \frac{
            (1 + \kappa \|y\|^2_2) x + (1 + \kappa \|x\|_2^2) y
            }{
            1 + \kappa^2 \|x\|^2_2 \|y\|^2_2
        },

    where :math:`\operatorname{gyr}[a, b]v = \ominus_\kappa (a \oplus_\kappa b)
    \oplus_\kappa (a \oplus_\kappa (b \oplus_\kappa v))`

    The following right cancellation property holds

    .. math::

        (a \boxplus_\kappa b) \ominus_\kappa b = a\\
        (a \oplus_\kappa b) \boxminus_\kappa b = a

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius coaddition

    """
    return _mobius_coadd(x, y, k, dim=dim)


# TODO: check numerical stability with Gregor's paper!!!
@torch.jit.script
def _mobius_coadd(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    # x2 = x.pow(2).sum(dim=dim, keepdim=True)
    # y2 = y.pow(2).sum(dim=dim, keepdim=True)
    # num = (1 + K * y2) * x + (1 + K * x2) * y
    # denom = 1 - K ** 2 * x2 * y2
    # avoid division by zero in this way
    # return num / denom.clamp_min(1e-15)
    #
    return _mobius_add(x, _gyration(x, -y, y, k=k, dim=dim), k, dim=dim)


def mobius_cosub(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius gyrovector cosubtraction.

    The Möbius cosubtraction is defined as follows:

    .. math::

        a \boxminus_\kappa b = a \boxplus_\kappa -b

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius cosubtraction

    """
    return _mobius_cosub(x, y, k, dim=dim)


@torch.jit.script
def _mobius_cosub(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    return _mobius_coadd(x, -y, k, dim=dim)


# TODO: can we make this operation somehow safer by breaking up the
# TODO: scalar multiplication for K>0 when the argument to the
# TODO: tan function gets close to pi/2+k*pi for k in Z?
# TODO: one could use the scalar associative law
# TODO: s_1 (X) s_2 (X) x = (s_1*s_2) (X) x
# TODO: to implement a more stable Möbius scalar mult
def mobius_scalar_mul(r: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the Möbius scalar multiplication.

    .. math::

        r \otimes_\kappa x
        =
        \tan_\kappa(r\tan_\kappa^{-1}(\|x\|_2))\frac{x}{\|x\|_2}

    This operation has properties similar to the Euclidean scalar multiplication

    * `n-addition` property

    .. math::

         r \otimes_\kappa x = x \oplus_\kappa \dots \oplus_\kappa x

    * Distributive property

    .. math::

         (r_1 + r_2) \otimes_\kappa x
         =
         r_1 \otimes_\kappa x \oplus r_2 \otimes_\kappa x

    * Scalar associativity

    .. math::

         (r_1 r_2) \otimes_\kappa x = r_1 \otimes_\kappa (r_2 \otimes_\kappa x)

    * Monodistributivity

    .. math::

         r \otimes_\kappa (r_1 \otimes x \oplus r_2 \otimes x) =
         r \otimes_\kappa (r_1 \otimes x) \oplus r \otimes (r_2 \otimes x)

    * Scaling property

    .. math::

        |r| \otimes_\kappa x / \|r \otimes_\kappa x\|_2 = x/\|x\|_2

    Parameters
    ----------
    r : tensor
        scalar for multiplication
    x : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius scalar multiplication
    """
    return _mobius_scalar_mul(r, x, k, dim=dim)


@torch.jit.script
def _mobius_scalar_mul(
    r: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = tan_k(r * artan_k(x_norm, k), k) * (x / x_norm)
    return res_c


def dist(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, keepdim=False, dim=-1):
    r"""
    Compute the geodesic distance between :math:`x` and :math:`y` on the manifold.

    .. math::

        d_\kappa(x, y) = 2\tan_\kappa^{-1}(\|(-x)\oplus_\kappa y\|_2)

    .. plot:: plots/extended/stereographic/distance.py

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    return _dist(x, y, k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _dist(
    x: torch.Tensor,
    y: torch.Tensor,
    k: torch.Tensor,
    keepdim: bool = False,
    dim: int = -1,
):
    return 2.0 * artan_k(
        _mobius_add(-x, y, k, dim=dim).norm(dim=dim, p=2, keepdim=keepdim), k
    )


def dist0(x: torch.Tensor, *, k: torch.Tensor, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance to the manifold's origin.

    Parameters
    ----------
    x : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`0`
    """
    return _dist0(x, k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _dist0(x: torch.Tensor, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    return 2.0 * artan_k(x.norm(dim=dim, p=2, keepdim=keepdim), k)


def geodesic(
    t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1
):
    r"""
    Compute the point on the path connecting :math:`x` and :math:`y` at time :math:`x`.

    The path can also be treated as an extension of the line segment to an
    unbounded geodesic that goes through :math:`x` and :math:`y`. The equation
    of the geodesic is given as:

    .. math::

        \gamma_{x\to y}(t)
        =
        x \oplus_\kappa t \otimes_\kappa ((-x) \oplus_\kappa y)

    The properties of the geodesic are the following:

    .. math::

        \gamma_{x\to y}(0) = x\\
        \gamma_{x\to y}(1) = y\\
        \dot\gamma_{x\to y}(t) = v

    Furthermore, the geodesic also satisfies the property of local distance
    minimization:

    .. math::

         d_\kappa(\gamma_{x\to y}(t_1), \gamma_{x\to y}(t_2)) = v|t_1-t_2|

    "Natural parametrization" of the curve ensures unit speed geodesics which
    yields the above formula with :math:`v=1`.

    However, we can always compute the constant speed :math:`v` from the points
    that the particular path connects:

    .. math::

        v = d_\kappa(\gamma_{x\to y}(0), \gamma_{x\to y}(1)) = d_\kappa(x, y)


    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        point on the geodesic going through x and y
    """
    return _geodesic(t, x, y, k, dim=dim)


@torch.jit.script
def _geodesic(
    t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    # this is not very numerically stable
    v = _mobius_add(-x, y, k, dim=dim)
    tv = _mobius_scalar_mul(t, v, k, dim=dim)
    gamma_t = _mobius_add(x, tv, k, dim=dim)
    return gamma_t


def expmap(x: torch.Tensor, u: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the exponential map of :math:`u` at :math:`x`.

    The expmap is tightly related with :func:`geodesic`. Intuitively, the
    expmap represents a smooth travel along a geodesic from the starting point
    :math:`x`, into the initial direction :math:`u` at speed :math:`\|u\|_x` for
    the duration of one time unit. In formulas one can express this as the
    travel along the curve :math:`\gamma_{x, u}(t)` such that

    .. math::

        \gamma_{x, u}(0) = x\\
        \dot\gamma_{x, u}(0) = u\\
        \|\dot\gamma_{x, u}(t)\|_{\gamma_{x, u}(t)} = \|u\|_x

    The existence of this curve relies on uniqueness of the differential
    equation solution, that is local. For the universal manifold the solution
    is well defined globally and we have.

    .. math::

        \operatorname{exp}^\kappa_x(u) = \gamma_{x, u}(1) = \\
        x\oplus_\kappa \tan_\kappa(\|u\|_x/2) \frac{u}{\|u\|_2}

    Parameters
    ----------
    x : tensor
        starting point on manifold
    u : tensor
        speed vector in tangent space at x
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    return _expmap(x, u, k, dim=dim)


@torch.jit.script
def _expmap(x: torch.Tensor, u: torch.Tensor, k: torch.Tensor, dim: int = -1):
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    lam = _lambda_x(x, k, dim=dim, keepdim=True)
    second_term = tan_k((lam / 2.0) * u_norm, k) * (u / u_norm)
    y = _mobius_add(x, second_term, k, dim=dim)
    return y


def expmap0(u: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the exponential map of :math:`u` at the origin :math:`0`.

    .. math::

        \operatorname{exp}^\kappa_0(u)
        =
        \tan_\kappa(\|u\|_2/2) \frac{u}{\|u\|_2}

    Parameters
    ----------
    u : tensor
        speed vector on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    return _expmap0(u, k, dim=dim)


@torch.jit.script
def _expmap0(u: torch.Tensor, k: torch.Tensor, dim: int = -1):
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    gamma_1 = tan_k(u_norm, k) * (u / u_norm)
    return gamma_1


def geodesic_unit(
    t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, k: torch.Tensor, dim=-1
):
    r"""
    Compute the point on the unit speed geodesic.

    The point on the unit speed geodesic at time :math:`t`, starting
    from :math:`x` with initial direction :math:`u/\|u\|_x` is computed
    as follows:

    .. math::

        \gamma_{x,u}(t) = x\oplus_\kappa \tan_\kappa(t/2) \frac{u}{\|u\|_2}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point on manifold
    u : tensor
        initial direction in tangent space at x
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the point on the unit speed geodesic
    """
    return _geodesic_unit(t, x, u, k, dim=dim)


@torch.jit.script
def _geodesic_unit(
    t: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    k: torch.Tensor,
    dim: int = -1,
):
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    second_term = tan_k(t / 2.0, k) * (u / u_norm)
    gamma_1 = _mobius_add(x, second_term, k, dim=dim)
    return gamma_1


def logmap(x: torch.Tensor, y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the logarithmic map of :math:`y` at :math:`x`.

    .. math::

        \operatorname{log}^\kappa_x(y) = \frac{2}{\lambda_x^\kappa}
        \tan_\kappa^{-1}(\|(-x)\oplus_\kappa y\|_2)
        * \frac{(-x)\oplus_\kappa y}{\|(-x)\oplus_\kappa y\|_2}

    The result of the logmap is a vector :math:`u` in the tangent space of
    :math:`x` such that

    .. math::

        y = \operatorname{exp}^\kappa_x(\operatorname{log}^\kappa_x(y))


    Parameters
    ----------
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_x M` that transports :math:`x` to :math:`y`
    """
    return _logmap(x, y, k, dim=dim)


@torch.jit.script
def _logmap(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    sub = _mobius_add(-x, y, k, dim=dim)
    sub_norm = sub.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    lam = _lambda_x(x, k, keepdim=True, dim=dim)
    return 2.0 * artan_k(sub_norm, k) * (sub / (lam * sub_norm))


def logmap0(y: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the logarithmic map of :math:`y` at the origin :math:`0`.

    .. math::

        \operatorname{log}^\kappa_0(y)
        =
        \tan_\kappa^{-1}(\|y\|_2) \frac{y}{\|y\|_2}

    The result of the logmap at the origin is a vector :math:`u` in the tangent
    space of the origin :math:`0` such that

    .. math::

        y = \operatorname{exp}^\kappa_0(\operatorname{log}^\kappa_0(y))

    Parameters
    ----------
    y : tensor
        target point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_0 M` that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, k, dim=dim)


@torch.jit.script
def _logmap0(y: torch.Tensor, k, dim: int = -1):
    y_norm = y.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    return (y / y_norm) * artan_k(y_norm, k)


def mobius_matvec(m: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the generalization of matrix-vector multiplication in gyrovector spaces.

    The Möbius matrix vector operation is defined as follows:

    .. math::

        M \otimes_\kappa x = \tan_\kappa\left(
            \frac{\|Mx\|_2}{\|x\|_2}\tan_\kappa^{-1}(\|x\|_2)
        \right)\frac{Mx}{\|Mx\|_2}

    .. plot:: plots/extended/stereographic/mobius_matvec.py

    Parameters
    ----------
    m : tensor
        matrix for multiplication. Batched matmul is performed if
        ``m.dim() > 2``, but only last dim reduction is supported
    x : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Möbius matvec result
    """
    return _mobius_matvec(m, x, k, dim=dim)


@torch.jit.script
def _mobius_matvec(m: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    if m.dim() > 2 and dim != -1:
        raise RuntimeError(
            "broadcasted Möbius matvec is supported for the last dim only"
        )
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    if dim != -1 or m.dim() == 2:
        mx = torch.tensordot(x, m, ([dim], [1]))
    else:
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
    mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = tan_k(mx_norm / x_norm * artan_k(x_norm, k), k) * (mx / mx_norm)
    cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.bool)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res


# TODO: check if this extends to gyrovector spaces for positive curvature
# TODO: add plot
def mobius_pointwise_mul(w: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the generalization for point-wise multiplication in gyrovector spaces.

    The Möbius pointwise multiplication is defined as follows

    .. math::

        \operatorname{diag}(w) \otimes_\kappa x = \tan_\kappa\left(
            \frac{\|\operatorname{diag}(w)x\|_2}{x}\tanh^{-1}(\|x\|_2)
        \right)\frac{\|\operatorname{diag}(w)x\|_2}{\|x\|_2}


    Parameters
    ----------
    w : tensor
        weights for multiplication (should be broadcastable to x)
    x : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Möbius point-wise mul result
    """
    return _mobius_pointwise_mul(w, x, k, dim=dim)


@torch.jit.script
def _mobius_pointwise_mul(
    w: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    wx = w * x
    wx_norm = wx.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = tan_k(wx_norm / x_norm * artan_k(x_norm, k), k) * (wx / wx_norm)
    zero = torch.zeros((), dtype=res_c.dtype, device=res_c.device)
    cond = wx.isclose(zero).prod(dim=dim, keepdim=True, dtype=torch.bool)
    res = torch.where(cond, zero, res_c)
    return res


def mobius_fn_apply_chain(x: torch.Tensor, *fns: callable, k: torch.Tensor, dim=-1):
    r"""
    Compute the generalization of sequential function application in gyrovector spaces.

    First, a gyrovector is mapped to the tangent space (first-order approx.) via
    :math:`\operatorname{log}^\kappa_0` and then the sequence of functions is
    applied to the vector in the tangent space. The resulting tangent vector is
    then mapped back with :math:`\operatorname{exp}^\kappa_0`.

    .. math::

        f^{\otimes_\kappa}(x)
        =
        \operatorname{exp}^\kappa_0(f(\operatorname{log}^\kappa_0(y)))

    The definition of mobius function application allows chaining as

    .. math::

        y = \operatorname{exp}^\kappa_0(\operatorname{log}^\kappa_0(y))

    Resulting in

    .. math::

        (f \circ g)^{\otimes_\kappa}(x)
        =
        \operatorname{exp}^\kappa_0(
            (f \circ g) (\operatorname{log}^\kappa_0(y))
        )

    Parameters
    ----------
    x : tensor
        point on manifold
    fns : callable[]
        functions to apply
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Apply chain result
    """
    if not fns:
        return x
    else:
        ex = _logmap0(x, k, dim=dim)
        for fn in fns:
            ex = fn(ex)
        y = _expmap0(ex, k, dim=dim)
        return y


def mobius_fn_apply(
    fn: callable, x: torch.Tensor, *args, k: torch.Tensor, dim=-1, **kwargs
):
    r"""
    Compute the generalization of function application in gyrovector spaces.

    First, a gyrovector is mapped to the tangent space (first-order approx.) via
    :math:`\operatorname{log}^\kappa_0` and then the function is applied
    to the vector in the tangent space. The resulting tangent vector is then
    mapped back with :math:`\operatorname{exp}^\kappa_0`.

    .. math::

        f^{\otimes_\kappa}(x)
        =
        \operatorname{exp}^\kappa_0(f(\operatorname{log}^\kappa_0(y)))

    .. plot:: plots/extended/stereographic/mobius_sigmoid_apply.py

    Parameters
    ----------
    x : tensor
        point on manifold
    fn : callable
        function to apply
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Result of function in hyperbolic space
    """
    ex = _logmap0(x, k, dim=dim)
    ex = fn(ex, *args, **kwargs)
    y = _expmap0(ex, k, dim=dim)
    return y


def mobiusify(fn: callable):
    r"""
    Wrap a function such that is works in gyrovector spaces.

    Parameters
    ----------
    fn : callable
        function in Euclidean space

    Returns
    -------
    callable
        function working in gyrovector spaces

    Notes
    -----
    New function will accept additional argument ``k`` and ``dim``.
    """

    @functools.wraps(fn)
    def mobius_fn(x, *args, k, dim=-1, **kwargs):
        ex = _logmap0(x, k, dim=dim)
        ex = fn(ex, *args, **kwargs)
        y = _expmap0(ex, k, dim=dim)
        return y

    return mobius_fn


def dist2plane(
    x: torch.Tensor,
    p: torch.Tensor,
    a: torch.Tensor,
    *,
    k: torch.Tensor,
    keepdim=False,
    signed=False,
    scaled=False,
    dim=-1,
):
    r"""
    Geodesic distance from :math:`x` to a hyperplane :math:`H_{a, b}`.

    The hyperplane is such that its set of points is orthogonal to :math:`a` and
    contains :math:`p`.

    .. plot:: plots/extended/stereographic/distance2plane.py

    To form an intuition what is a hyperplane in gyrovector spaces, let's first
    consider an Euclidean hyperplane

    .. math::

        H_{a, b} = \left\{
            x \in \mathbb{R}^n\;:\;\langle x, a\rangle - b = 0
        \right\},

    where :math:`a\in \mathbb{R}^n\backslash \{\mathbf{0}\}` and
    :math:`b\in \mathbb{R}^n`.

    This formulation of a hyperplane is hard to generalize,
    therefore we can rewrite :math:`\langle x, a\rangle - b`
    utilizing orthogonal completion.
    Setting any :math:`p` s.t. :math:`b=\langle a, p\rangle` we have

    .. math::

        H_{a, b} = \left\{
            x \in \mathbb{R}^n\;:\;\langle x, a\rangle - b = 0
        \right\}\\
        =H_{a, \langle a, p\rangle} = \tilde{H}_{a, p}\\
        = \left\{
            x \in \mathbb{R}^n\;:\;\langle x, a\rangle - \langle a, p\rangle = 0
        \right\}\\
        =\left\{
            x \in \mathbb{R}^n\;:\;\langle -p + x, a\rangle = 0
        \right\}\\
        = p + \{a\}^\perp

    Naturally we have a set :math:`\{a\}^\perp` with applied :math:`+` operator
    to each element. Generalizing a notion of summation to the gyrovector space
    we replace :math:`+` with :math:`\oplus_\kappa`.

    Next, we should figure out what is :math:`\{a\}^\perp` in the gyrovector
    space.

    First thing that we should acknowledge is that notion of orthogonality is
    defined for vectors in tangent spaces. Let's consider now
    :math:`p\in \mathcal{M}_\kappa^n` and
    :math:`a\in T_p\mathcal{M}_\kappa^n\backslash \{\mathbf{0}\}`.

    Slightly deviating from traditional notation let's write
    :math:`\{a\}_p^\perp` highlighting the tight relationship of
    :math:`a\in T_p\mathcal{M}_\kappa^n\backslash \{\mathbf{0}\}`
    with :math:`p \in \mathcal{M}_\kappa^n`. We then define

    .. math::

        \{a\}_p^\perp := \left\{
            z\in T_p\mathcal{M}_\kappa^n \;:\; \langle z, a\rangle_p = 0
        \right\}

    Recalling that a tangent vector :math:`z` for point :math:`p` yields
    :math:`x = \operatorname{exp}^\kappa_p(z)` we rewrite the above equation as

    .. math::
        \{a\}_p^\perp := \left\{
            x\in \mathcal{M}_\kappa^n \;:\; \langle
            \operatorname{log}_p^\kappa(x), a\rangle_p = 0
        \right\}

    This formulation is something more pleasant to work with.
    Putting all together

    .. math::

        \tilde{H}_{a, p}^\kappa = p + \{a\}^\perp_p\\
        = \left\{
            x \in \mathcal{M}_\kappa^n\;:\;\langle
            \operatorname{log}^\kappa_p(x),
            a\rangle_p = 0
        \right\} \\
        = \left\{
            x \in \mathcal{M}_\kappa^n\;:\;\langle -p \oplus_\kappa x, a\rangle
            = 0
        \right\}

    To compute the distance :math:`d_\kappa(x, \tilde{H}_{a, p}^\kappa)` we find

    .. math::

        d_\kappa(x, \tilde{H}_{a, p}^\kappa)
        =
        \inf_{w\in \tilde{H}_{a, p}^\kappa} d_\kappa(x, w)\\
        =
        \sin^{-1}_\kappa\left\{
            \frac{
            2 |\langle(-p)\oplus_\kappa x, a\rangle|
            }{
            (1+\kappa\|(-p)\oplus_\kappa \|x\|^2_2)\|a\|_2
            }
        \right\}

    Parameters
    ----------
    x : tensor
        point on manifold to compute distance for
    a : tensor
        hyperplane normal vector in tangent space of :math:`p`
    p : tensor
        point on manifold lying on the hyperplane
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    signed : bool
        return signed distance
    scaled : bool
        scale distance by tangent norm
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        distance to the hyperplane
    """
    return _dist2plane(
        x, a, p, k, keepdim=keepdim, signed=signed, dim=dim, scaled=scaled
    )


@torch.jit.script
def _dist2plane(
    x: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
    keepdim: bool = False,
    signed: bool = False,
    scaled: bool = False,
    dim: int = -1,
):
    diff = _mobius_add(-p, x, k, dim=dim)
    diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(1e-15)
    sc_diff_a = (diff * a).sum(dim=dim, keepdim=keepdim)
    if not signed:
        sc_diff_a = sc_diff_a.abs()
    a_norm = a.norm(dim=dim, keepdim=keepdim, p=2)
    num = 2.0 * sc_diff_a
    denom = clamp_abs((1 + k * diff_norm2) * a_norm)
    distance = arsin_k(num / denom, k)
    if scaled:
        distance = distance * a_norm
    return distance


def parallel_transport(
    x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, k: torch.Tensor, dim=-1
):
    r"""
    Compute the parallel transport of :math:`v` from :math:`x` to :math:`y`.

    The parallel transport is essential for adaptive algorithms on Riemannian
    manifolds. For gyrovector spaces the parallel transport is expressed through
    the gyration.

    .. plot:: plots/extended/stereographic/gyrovector_parallel_transport.py

    To recover parallel transport we first need to study isomorphisms between
    gyrovectors and vectors. The reason is that originally, parallel transport
    is well defined for gyrovectors as

    .. math::

        P_{x\to y}(z) = \operatorname{gyr}[y, -x]z,

    where :math:`x,\:y,\:z \in \mathcal{M}_\kappa^n` and
    :math:`\operatorname{gyr}[a, b]c = \ominus (a \oplus_\kappa b)
    \oplus_\kappa (a \oplus_\kappa (b \oplus_\kappa c))`

    But we want to obtain parallel transport for vectors, not for gyrovectors.
    The blessing is the isomorphism mentioned above. This mapping is given by

    .. math::

        U^\kappa_p \: : \: T_p\mathcal{M}_\kappa^n \to \mathbb{G}
        =
        v \mapsto \lambda^\kappa_p v


    Finally, having the points :math:`x,\:y \in \mathcal{M}_\kappa^n` and a
    tangent vector :math:`u\in T_x\mathcal{M}_\kappa^n` we obtain

    .. math::

        P^\kappa_{x\to y}(v)
        =
        (U^\kappa_y)^{-1}\left(\operatorname{gyr}[y, -x] U^\kappa_x(v)\right)\\
        =
        \operatorname{gyr}[y, -x] v \lambda^\kappa_x / \lambda^\kappa_y

    .. plot:: plots/extended/stereographic/parallel_transport.py


    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector at x to be transported to y
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport(x, y, v, k, dim=dim)


@torch.jit.script
def _parallel_transport(
    x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    return (
        _gyration(y, -x, u, k, dim=dim)
        * _lambda_x(x, k, keepdim=True, dim=dim)
        / _lambda_x(y, k, keepdim=True, dim=dim)
    )


def parallel_transport0(y: torch.Tensor, v: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Compute the parallel transport of :math:`v` from the origin :math:`0` to :math:`y`.

    This is just a special case of the parallel transport with the starting
    point at the origin that can be computed more efficiently and more
    numerically stable.

    Parameters
    ----------
    y : tensor
        target point
    v : tensor
        vector to be transported from the origin to y
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0(y, v, k, dim=dim)


@torch.jit.script
def _parallel_transport0(
    y: torch.Tensor, v: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    return v * (1 + k * y.pow(2).sum(dim=dim, keepdim=True)).clamp_min(1e-15)


def parallel_transport0back(
    x: torch.Tensor, v: torch.Tensor, *, k: torch.Tensor, dim: int = -1
):
    r"""
    Perform parallel transport to the zero point.

    Special case parallel transport with last point at zero that
    can be computed more efficiently and numerically stable

    Parameters
    ----------
    x : tensor
        target point
    v : tensor
        vector to be transported
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0back(x, v, k=k, dim=dim)


@torch.jit.script
def _parallel_transport0back(
    x: torch.Tensor, v: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    return v / (1 + k * x.pow(2).sum(dim=dim, keepdim=True)).clamp_min(1e-15)


def egrad2rgrad(x: torch.Tensor, grad: torch.Tensor, *, k: torch.Tensor, dim=-1):
    r"""
    Convert the Euclidean gradient to the Riemannian gradient.

    .. math::

        \nabla_x = \nabla^E_x / (\lambda_x^\kappa)^2

    Parameters
    ----------
    x : tensor
        point on the manifold
    grad : tensor
        Euclidean gradient for :math:`x`
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in T_x\mathcal{M}_\kappa^n`
    """
    return _egrad2rgrad(x, grad, k, dim=dim)


@torch.jit.script
def _egrad2rgrad(x: torch.Tensor, grad: torch.Tensor, k: torch.Tensor, dim: int = -1):
    return grad / _lambda_x(x, k, keepdim=True, dim=dim) ** 2


def sproj(x: torch.Tensor, *, k: torch.Tensor, dim: int = -1):
    """
    Stereographic Projection from hyperboloid or sphere.

    Parameters
    ----------
    x : tensor
        point to be projected
    k : tensor
        constant sectional curvature
    dim : int
        dimension to operate on

    Returns
    -------
    tensor
        the result of the projection
    """
    return _sproj(x, k, dim=dim)


@torch.jit.script
def _sproj(x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    inv_r = torch.sqrt(sabs(k))
    factor = 1.0 / (1.0 + inv_r * x.narrow(dim, -1, 1))
    proj = factor * x.narrow(dim, 0, x.size(dim) - 1)
    return proj


def inv_sproj(x: torch.Tensor, *, k: torch.Tensor, dim: int = -1):
    """
    Inverse of Stereographic Projection to hyperboloid or sphere.

    Parameters
    ----------
    x : tensor
        point to be projected
    k : tensor
        constant sectional curvature
    dim : int
        dimension to operate on

    Returns
    -------
    tensor
        the result of the projection
    """
    return _inv_sproj(x, k, dim=dim)


@torch.jit.script
def _inv_sproj(x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    inv_r = torch.sqrt(sabs(k))
    lam_x = _lambda_x(x, k, keepdim=True, dim=dim)
    A = lam_x * x
    B = 1.0 / inv_r * (lam_x - 1.0)
    proj = torch.cat((A, B), dim=dim)
    return proj


def antipode(x: torch.Tensor, *, k: torch.Tensor, dim: int = -1):
    r"""
    Compute the antipode of a point :math:`x_1,...,x_n` for :math:`\kappa > 0`.

    Let :math:`x` be a point on some sphere. Then :math:`-x` is its antipode.
    Since we're dealing with stereographic projections, for :math:`sproj(x)` we
    get the antipode :math:`sproj(-x)`. Which is given as follows:

    .. math::

        \text{antipode}(x)
        =
        \frac{1+\kappa\|x\|^2_2}{2\kappa\|x\|^2_2}{}(-x)

    Parameters
    ----------
    x : tensor
        points :math:`x_1,...,x_n` on manifold to compute antipode for
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        antipode
    """
    return _antipode(x, k, dim=dim)


@torch.jit.script
def _antipode(x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    # NOTE: implementation that uses stereographic projections seems to be less accurate
    # sproj(-inv_sproj(x))
    if torch.all(k.le(0)):
        return -x
    v = x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(1e-15)
    R = sabs(k).sqrt().reciprocal()
    pi = 3.141592653589793

    a = _geodesic_unit(pi * R, x, v, k, dim=dim)
    return torch.where(k.gt(0), a, -x)


def weighted_midpoint(
    xs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    k: torch.Tensor,
    reducedim: Optional[List[int]] = None,
    dim: int = -1,
    keepdim: bool = False,
    lincomb: bool = False,
    posweight: bool = False,
):
    r"""
    Compute weighted Möbius gyromidpoint.

    The weighted Möbius gyromidpoint of a set of points
    :math:`x_1,...,x_n` according to weights
    :math:`\alpha_1,...,\alpha_n` is computed as follows:

    The weighted Möbius gyromidpoint is computed as follows

    .. math::

        m_{\kappa}(x_1,\ldots,x_n,\alpha_1,\ldots,\alpha_n)
        =
        \frac{1}{2}
        \otimes_\kappa
        \left(
        \sum_{i=1}^n
        \frac{
        \alpha_i\lambda_{x_i}^\kappa
        }{
        \sum_{j=1}^n\alpha_j(\lambda_{x_j}^\kappa-1)
        }
        x_i
        \right)

    where the weights :math:`\alpha_1,...,\alpha_n` do not necessarily need
    to sum to 1 (only their relative weight matters). Note that this formula
    also requires to choose between the midpoint and its antipode for
    :math:`\kappa > 0`.

    Parameters
    ----------
    xs : tensor
        points on poincare ball
    weights : tensor
        weights for averaging (make sure they broadcast correctly and manifold dimension is skipped)
    reducedim : int|list|tuple
        reduce dimension
    dim : int
        dimension to calculate conformal and Lorenz factors
    k : tensor
        constant sectional curvature
    keepdim : bool
        retain the last dim? (default: false)
    lincomb : bool
        linear combination implementation
    posweight : bool
        make all weights positive. Negative weight will weight antipode of entry with positive weight instead.
        This will give experimentally better numerics and nice interpolation
        properties for linear combination and averaging

    Returns
    -------
    tensor
        Einstein midpoint in poincare coordinates
    """
    return _weighted_midpoint(
        xs=xs,
        k=k,
        weights=weights,
        reducedim=reducedim,
        dim=dim,
        keepdim=keepdim,
        lincomb=lincomb,
        posweight=posweight,
    )


@torch.jit.script
def _weighted_midpoint(
    xs: torch.Tensor,
    k: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    reducedim: Optional[List[int]] = None,
    dim: int = -1,
    keepdim: bool = False,
    lincomb: bool = False,
    posweight: bool = False,
):
    if reducedim is None:
        reducedim = list_range(xs.dim())
        reducedim.pop(dim)
    gamma = _lambda_x(xs, k=k, dim=dim, keepdim=True)
    if weights is None:
        weights = torch.tensor(1.0, dtype=xs.dtype, device=xs.device)
    else:
        weights = weights.unsqueeze(dim)
    if posweight and weights.lt(0).any():
        xs = torch.where(weights.lt(0), _antipode(xs, k=k, dim=dim), xs)
        weights = weights.abs()
    denominator = ((gamma - 1) * weights).sum(reducedim, keepdim=True)
    nominator = (gamma * weights * xs).sum(reducedim, keepdim=True)
    two_mean = nominator / clamp_abs(denominator, 1e-10)
    a_mean = _mobius_scalar_mul(
        torch.tensor(0.5, dtype=xs.dtype, device=xs.device), two_mean, k=k, dim=dim
    )
    if torch.any(k.gt(0)):
        # check antipode
        b_mean = _antipode(a_mean, k, dim=dim)
        a_dist = _dist(a_mean, xs, k=k, keepdim=True, dim=dim).sum(
            reducedim, keepdim=True
        )
        b_dist = _dist(b_mean, xs, k=k, keepdim=True, dim=dim).sum(
            reducedim, keepdim=True
        )
        better = k.gt(0) & (b_dist < a_dist)
        a_mean = torch.where(better, b_mean, a_mean)
    if lincomb:
        if weights.numel() == 1:
            alpha = weights.clone()
            for d in reducedim:
                alpha *= xs.size(d)
        else:
            weights, _ = torch.broadcast_tensors(weights, gamma)
            alpha = weights.sum(reducedim, keepdim=True)
        a_mean = _mobius_scalar_mul(alpha, a_mean, k=k, dim=dim)
    if not keepdim:
        a_mean = drop_dims(a_mean, reducedim)
    return a_mean
