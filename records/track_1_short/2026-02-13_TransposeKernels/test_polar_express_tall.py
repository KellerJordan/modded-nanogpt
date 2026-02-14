"""
Test script to verify numerical equivalence between:
1. Wide polar express: X @ X.T with left multiplication (original)
2. Tall polar express: X.T @ X with right multiplication (new)

Both should produce the same orthogonal polar factor.
"""
import torch

# Polar Express coefficients (same for both variants)
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def polar_express_wide(X: torch.Tensor) -> torch.Tensor:
    """
    Wide variant: X @ X.T with left multiplication.
    Works best when rows <= cols.
    """
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    for a, b, c in polar_express_coeffs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


def polar_express_tall(X: torch.Tensor) -> torch.Tensor:
    """
    Tall variant: X.T @ X with right multiplication.
    Works best when rows > cols.
    """
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    for a, b, c in polar_express_coeffs:
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B
    return X


def polar_express_with_transpose(X: torch.Tensor) -> torch.Tensor:
    """
    Original approach: transpose tall matrices, run wide variant, transpose back.
    This is what we want to replace for tall matrices.
    """
    is_tall = X.size(-2) > X.size(-1)
    if is_tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    for a, b, c in polar_express_coeffs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if is_tall:
        X = X.mT
    return X


def test_equivalence():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print(f"Testing on device: {device}, dtype: {dtype}")
    print("=" * 60)
    
    # Test cases: (batch, rows, cols)
    test_cases = [
        # Wide matrices (rows <= cols) - for reference
        (1, 768, 768),      # Square
        (1, 768, 3072),     # Wide (attention-like)
        (5, 768, 768),      # Batched square
        
        # Tall matrices (rows > cols) - the focus
        (1, 3072, 768),     # Tall (MLP-like)
        (3, 3072, 768),     # Batched tall
        (24, 3072, 768),    # Full MLP batch size
    ]
    
    for batch, rows, cols in test_cases:
        if batch == 1:
            X = torch.randn(rows, cols, device=device, dtype=dtype)
        else:
            X = torch.randn(batch, rows, cols, device=device, dtype=dtype)
        
        is_tall = rows > cols
        shape_str = f"{'Tall' if is_tall else 'Wide/Sq':>7}"
        
        # Run both variants
        if is_tall:
            # For tall matrices, compare tall variant with transpose approach
            result_tall = polar_express_tall(X.clone())
            result_transpose = polar_express_with_transpose(X.clone())
            
            # Compute difference
            diff = (result_tall - result_transpose).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            status = "✓ PASS" if max_diff < 0.1 else "✗ FAIL"
            print(f"{status} {shape_str} shape={str(tuple(X.shape)):20} | "
                  f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}")
        else:
            # For wide/square matrices, just verify wide variant works
            result_wide = polar_express_wide(X.clone())
            result_transpose = polar_express_with_transpose(X.clone())
            
            diff = (result_wide - result_transpose).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            status = "✓ PASS" if max_diff < 0.1 else "✗ FAIL"
            print(f"{status} {shape_str} shape={str(tuple(X.shape))[:20]} | "
                  f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}")
    
    print("=" * 60)
    print("\nNow testing Triton kernels...")
    test_triton_kernels()


def test_triton_kernels():
    """Test the Triton kernel implementations."""
    from triton_kernels import XXT, ba_plus_cAA
    
    # Try to import XTX if it exists
    try:
        from triton_kernels import XTX
        has_xtx = True
    except ImportError:
        has_xtx = False
        print("XTX kernel not yet implemented, skipping Triton tall tests")
    
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    
    print("\n--- Testing XXT (wide) kernel ---")
    # Test XXT kernel matches torch
    for batch, rows, cols in [(1, 768, 3072), (5, 768, 768), (24, 768, 3072)]:
        if batch == 1:
            X = torch.randn(rows, cols, device=device, dtype=dtype)
            A_torch = X @ X.mT
            A_triton = torch.empty(rows, rows, device=device, dtype=dtype)
        else:
            X = torch.randn(batch, rows, cols, device=device, dtype=dtype)
            A_torch = X @ X.mT
            A_triton = torch.empty(batch, rows, rows, device=device, dtype=dtype)
        
        XXT(X, out=A_triton)
        
        diff = (A_torch - A_triton).abs()
        max_diff = diff.max().item()
        status = "✓" if max_diff < 0.1 else "✗"
        print(f"  {status} XXT shape={tuple(X.shape)} -> {tuple(A_triton.shape)} | max_diff={max_diff:.6f}")
    
    if has_xtx:
        print("\n--- Testing XTX (tall) kernel ---")
        # Test XTX kernel matches torch
        for batch, rows, cols in [(1, 3072, 768), (3, 3072, 768), (24, 3072, 768)]:
            if batch == 1:
                X = torch.randn(rows, cols, device=device, dtype=dtype)
                A_torch = X.mT @ X
                A_triton = torch.empty(cols, cols, device=device, dtype=dtype)
            else:
                X = torch.randn(batch, rows, cols, device=device, dtype=dtype)
                A_torch = X.mT @ X
                A_triton = torch.empty(batch, cols, cols, device=device, dtype=dtype)
            
            XTX(X, out=A_triton)
            
            diff = (A_torch - A_triton).abs()
            max_diff = diff.max().item()
            status = "✓" if max_diff < 0.1 else "✗"
            print(f"  {status} XTX shape={tuple(X.shape)} -> {tuple(A_triton.shape)} | max_diff={max_diff:.6f}")
        
        print("\n--- Testing full tall polar express with Triton ---")
        test_full_triton_tall()


def test_full_triton_tall():
    """Test the full tall polar express using Triton kernels."""
    try:
        from triton_kernels import XTX, ba_plus_cAA
    except ImportError:
        print("  XTX not available, skipping")
        return
    
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    
    def polar_express_tall_triton(X: torch.Tensor) -> torch.Tensor:
        """Tall variant using Triton kernels."""
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        X = X.contiguous()
        
        # Allocate buffers
        if X.ndim == 2:
            A = torch.empty(X.size(-1), X.size(-1), device=X.device, dtype=X.dtype)
        else:
            A = torch.empty(X.size(0), X.size(-1), X.size(-1), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)
        
        for a, b, c in polar_express_coeffs:
            XTX(X, out=A)  # A = X.T @ X
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b*A + c*(A@A)
            
            # C = a*X + X @ B
            if X.ndim > 2:
                torch.bmm(X, B, out=C)
            else:
                torch.mm(X, B, out=C)
            C.add_(X, alpha=a)
            
            X, C = C, X
        
        return X
    
    for batch, rows, cols in [(1, 3072, 768), (3, 3072, 768), (24, 3072, 768)]:
        if batch == 1:
            X = torch.randn(rows, cols, device=device, dtype=dtype)
        else:
            X = torch.randn(batch, rows, cols, device=device, dtype=dtype)
        
        result_pytorch = polar_express_tall(X.clone())
        result_triton = polar_express_tall_triton(X.clone())
        
        diff = (result_pytorch - result_triton).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        status = "✓ PASS" if max_diff < 0.1 else "✗ FAIL"
        print(f"  {status} Full tall PE shape={tuple(X.shape)} | "
              f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}")


if __name__ == "__main__":
    test_equivalence()
