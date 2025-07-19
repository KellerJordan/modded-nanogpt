"""
Newton-Schulz orthogonalization with CUDA acceleration

This module provides a drop-in replacement for the zeropower_via_newtonschulz5 function
with an optimized CUDA kernel implementation.
"""

import torch
import warnings

# Try to import the CUDA extension
try:
    import newtonschulz5_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("CUDA extension for Newton-Schulz not found. Falling back to PyTorch implementation.")

@torch.compile
def zeropower_via_newtonschulz5_pytorch(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Original PyTorch implementation of Newton-Schulz iteration.
    Used as fallback when CUDA kernel is not available.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    
    This function automatically uses the CUDA kernel if available and the input
    is on a CUDA device, otherwise falls back to the PyTorch implementation.
    
    Args:
        G: Input tensor of shape (..., M, N)
        steps: Number of Newton-Schulz iterations
        
    Returns:
        Orthogonalized tensor of the same shape as G
    """
    if CUDA_AVAILABLE and G.is_cuda and G.is_contiguous():
        try:
            # Use CUDA kernel
            return newtonschulz5_cuda.newtonschulz5(G, steps)
        except Exception as e:
            warnings.warn(f"CUDA kernel failed: {e}. Falling back to PyTorch implementation.")
            return zeropower_via_newtonschulz5_pytorch(G, steps)
    else:
        # Use PyTorch implementation
        return zeropower_via_newtonschulz5_pytorch(G, steps)

# For backward compatibility, also export the original name
zeropower_via_newtonschulz5_cuda = zeropower_via_newtonschulz5