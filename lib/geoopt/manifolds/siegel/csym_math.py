import torch
from typing import Tuple
from lib.geoopt.linalg import block_matrix

EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

to_complex = torch.complex
inverse = torch.linalg.inv
eigh = torch.linalg.eigh
eigvalsh = torch.linalg.eigvalsh


def takagi_eig(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute the Takagi Factorization to obtain eigenvalues and eigenvectors.

    Given a complex symmetric matrix A, the Takagi factorization is an algorithm that
    computes a real diagonal matrix D and a complex unitary matrix S such that:

    .. math::

        A = \overline{S} D S^*

    where:
        - :math:`D` is a real nonnegative diagonal matrix
        - :math:`S` is complex unitary
        - :math:`\overline{S}`: S conjugate
        - :math:`S^*`: S conjugate transpose

    References
    ----------
    - https://en.wikipedia.org/wiki/Matrix_decomposition#Takagi's_factorization
    - https://github.com/hajifkd/takagi_fact/blob/master/takagi_fact/__init__.py

    Parameters
    ----------
    z : torch.Tensor
         Complex symmetric matrix

    Returns
    -------
    torch.Tensor
        Real eigenvalues of z
    torch.Tensor
        Complex eigenvectors of z
    """
    compound_z = to_compound_symmetric(
        z
    )  # Z = A + iB, then compound_z = [(A, B),(B, -A)]

    evalues, q = eigh(compound_z)  # evalues in ascending order

    # Think of Q as 4 block matrices.
    # Q = [(X,  Re(U)),
    #      (Y, -Im(U))]     where X, Y are irrelevant and I need to build U
    real_u_on_top_of_minus_imag_u = torch.chunk(q, 2, dim=-1)[-1]
    real_u, minus_imag_u = torch.chunk(real_u_on_top_of_minus_imag_u, 2, dim=-2)
    u = to_complex(real_u, -minus_imag_u)

    index = torch.arange(
        start=z.shape[-1], end=2 * z.shape[-1], device=evalues.device, dtype=torch.long
    )
    sing_values = evalues.index_select(dim=-1, index=index)
    return sing_values, u


def takagi_eigvals(z: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the Takagi Factorization to obtain eigenvalues and eigenvectors.

    Given a complex symmetric matrix A, the Takagi factorization is an algorithm that
    computes a real diagonal matrix D and a complex unitary matrix S such that:

    .. math::

        A = \overline{S} D S^*

    where:
        - :math:`D` is a real nonnegative diagonal matrix
        - :math:`S` is complex unitary
        - :math:`\overline{S}`: S conjugate
        - :math:`S^*`: S conjugate transpose

    References
    ----------
    - https://en.wikipedia.org/wiki/Matrix_decomposition#Takagi's_factorization
    - https://github.com/hajifkd/takagi_fact/blob/master/takagi_fact/__init__.py

    Parameters
    ----------
    z : torch.Tensor
         Complex symmetric matrix

    Returns
    -------
    torch.Tensor
        eigenvalues of z
    """
    compound_z = to_compound_symmetric(
        z
    )  # Z = A + iB, then compound_z = [(A, B),(B, -A)]
    evalues = eigvalsh(compound_z)  # evalues in ascending order
    index = torch.arange(
        start=z.shape[-1], end=2 * z.shape[-1], device=evalues.device, dtype=torch.long
    )
    sing_values = evalues.index_select(dim=-1, index=index)
    return sing_values


def cayley_transform(z: torch.Tensor) -> torch.Tensor:
    r"""
    Map elements from the Bounded Domain Model to the Upper Half Space Model.

    .. math::

        \operatorname{cayley}(Z): \mathcal{B}_n \to \mathcal{S}_n \\
        \operatorname{cayley}(Z) = i(Id + Z) (Id - Z)^{-1}

    References
    ----------
     - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map

    Parameters
    ----------
    z : torch.Tensor
         Point in the Bounded Domain model

    Returns
    -------
    torch.Tensor
        Point in the Upper Half Space model
    """
    identity = identity_like(z)

    i_z_plus_id = multiply_by_i(identity + z)
    inv_z_minus_id = inverse(identity - z)

    return i_z_plus_id @ inv_z_minus_id


def inverse_cayley_transform(z: torch.Tensor) -> torch.Tensor:
    r"""
    Map elements from the Upper Half Space Model to the Bounded Domain Model.

    .. math::

        \operatorname{cayley}^{-1}(X): \mathcal{S}_n \to \mathcal{B}_n \\
        \operatorname{cayley}^{-1}(X) = (X - i Id) (X + i Id)^{-1}

    References
    ----------
     - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map

    Parameters
    ----------
    z : torch.Tensor
         Point in the Upper Half Space model

    Returns
    -------
    torch.Tensor
        Point in the Bounded Domain model
    """
    identity = identity_like(z)
    i_identity = multiply_by_i(identity)

    z_minus_id = z - i_identity
    inv_z_plus_id = inverse(z + i_identity)
    return z_minus_id @ inv_z_plus_id


def is_complex_symmetric(z: torch.Tensor, atol=3e-5, rtol=1e-5):
    """
    Return whether the complex symmetric matrices are symmetric or not.

    Parameters
    ----------
    z : tensor
        Complex symmetric matrix
    atol : float
        absolute tolerance for allclose
    rtol : float
        relative tolerance for allclose

    Returns
    -------
    boolean
        whether the points in x are complex symmetric or not
    """
    real_z, imag_z = z.real, z.imag
    return torch.allclose(
        real_z, real_z.transpose(-1, -2), atol=atol, rtol=rtol
    ) and torch.allclose(imag_z, imag_z.transpose(-1, -2), atol=atol, rtol=rtol)


def to_compound_symmetric(z: torch.Tensor) -> torch.Tensor:
    """
    Build a real matrix out of the real and imaginary parts of a complex symmetric matrix.

    Let Z = A + iB be a matrix with complex entries, where A, B are n x n matrices with real entries.
    We build a 2n x 2n matrix in the following form:
        M = [(A, B),
             (B, -A)]
    Since Z is symmetric, then M is symmetric and with all real entries.

    Parameters
    ----------
    z : tensor
        Complex symmetric matrix of rank n

    Returns
    -------
    torch.Tensor
        Real symmetric matrix of rank 2n
    """
    a, b = z.real, z.imag
    return block_matrix([[a, b], [b, -a]])


def identity_like(z: torch.Tensor) -> torch.Tensor:
    """
    Return a complex identity of the shape of z, with the same type and device.

    Parameters
    ----------
    z : tensor
        Complex matrix of rank n

    Returns
    -------
    torch.Tensor
        Complex identity of rank n with zeros in the imaginary part
    """
    return torch.eye(z.shape[-1], dtype=z.dtype, device=z.device).expand_as(z)


def multiply_by_i(z: torch.Tensor):
    """
    For Z = X + iY, calculates the operation i Z = i (X + iY) = -Y + iX.

    Parameters
    ----------
    z : tensor
        Complex matrix

    Returns
    -------
    torch.Tensor
        Complex matrix
    """
    return to_complex(-z.imag, z.real)


def positive_conjugate_projection(y: torch.Tensor) -> torch.Tensor:
    """
    Apply a positive conjugate projection to a real symmetric matrix.

    Steps to project: Y = Symmetric Matrix
    1) Y = SDS^-1
    2) D_tilde = clamp(D, min=epsilon)
    3) Y_tilde = S D_tilde S^-1

    Parameters
    ----------
    y : torch.Tensor
         symmetric matrices
    """
    evalues, s = eigh(y)
    evalues = torch.clamp(evalues, min=EPS[y.dtype])
    y_tilde = s @ torch.diag_embed(evalues) @ s.transpose(-1, -2)

    # we do this so no operation is applied on the matrices that are already positive definite
    # This prevents modifying values due to numerical instabilities/floating point ops
    batch_wise_mask = torch.all(evalues > EPS[y.dtype], dim=-1, keepdim=True)
    mask = batch_wise_mask.unsqueeze(-1).expand_as(y)

    return torch.where(mask, y, y_tilde)
