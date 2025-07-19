"""
Test and benchmark the Newton-Schulz CUDA kernel implementation
"""

import torch
import time
import numpy as np
from newtonschulz5 import zeropower_via_newtonschulz5, zeropower_via_newtonschulz5_pytorch

def test_correctness():
    """Test that CUDA kernel produces similar results to PyTorch implementation"""
    print("Testing correctness...")
    
    # Test different tensor shapes and batch sizes
    test_cases = [
        (32, 64),      # rows < cols
        (64, 32),      # rows > cols  
        (64, 64),      # square
        (128, 256),    # larger
        (768, 768),    # typical transformer dimension
    ]
    
    for rows, cols in test_cases:
        print(f"\nTesting shape ({rows}, {cols}):")
        
        # Test without batch dimension
        G = torch.randn(rows, cols, dtype=torch.float32, device='cuda')
        
        # Run both implementations
        result_pytorch = zeropower_via_newtonschulz5_pytorch(G.clone(), steps=5)
        result_cuda = zeropower_via_newtonschulz5(G.clone(), steps=5)
        
        # Check that results are close
        max_diff = torch.max(torch.abs(result_pytorch - result_cuda)).item()
        mean_diff = torch.mean(torch.abs(result_pytorch - result_cuda)).item()
        
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        # Test orthogonality of result
        if rows <= cols:
            orth_error = torch.norm(result_cuda @ result_cuda.T - torch.eye(rows, device='cuda')).item()
        else:
            orth_error = torch.norm(result_cuda.T @ result_cuda - torch.eye(cols, device='cuda')).item()
        print(f"  Orthogonality error: {orth_error:.6f}")
        
        # Test with batch dimensions
        batch_G = torch.randn(4, 3, rows, cols, dtype=torch.float32, device='cuda')
        batch_result_pytorch = zeropower_via_newtonschulz5_pytorch(batch_G.clone(), steps=5)
        batch_result_cuda = zeropower_via_newtonschulz5(batch_G.clone(), steps=5)
        
        batch_max_diff = torch.max(torch.abs(batch_result_pytorch - batch_result_cuda)).item()
        print(f"  Batch max difference: {batch_max_diff:.6f}")
        
        assert max_diff < 1e-3, f"Large difference for shape ({rows}, {cols})"
        assert batch_max_diff < 1e-3, f"Large batch difference for shape ({rows}, {cols})"

def benchmark_performance():
    """Benchmark CUDA kernel vs PyTorch implementation"""
    print("\n\nBenchmarking performance...")
    
    # Warm up CUDA
    dummy = torch.randn(32, 32, device='cuda')
    _ = zeropower_via_newtonschulz5(dummy, steps=1)
    
    benchmark_cases = [
        ("Small (64x64)", 64, 64, 100),
        ("Medium (256x256)", 256, 256, 50),
        ("Large (768x768)", 768, 768, 20),
        ("Transformer (768x3072)", 768, 3072, 10),
        ("Batched (8x768x768)", (8, 768, 768), None, 10),
    ]
    
    for name, *shape_info in benchmark_cases:
        if len(shape_info) == 3:
            rows, cols, num_runs = shape_info
            shape = (rows, cols)
        else:
            shape, _, num_runs = shape_info
        
        print(f"\n{name}:")
        
        # Create test tensor
        G = torch.randn(*shape, dtype=torch.float32, device='cuda')
        
        # Benchmark PyTorch implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            _ = zeropower_via_newtonschulz5_pytorch(G.clone(), steps=5)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / num_runs * 1000
        
        # Benchmark CUDA kernel
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            _ = zeropower_via_newtonschulz5(G.clone(), steps=5)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / num_runs * 1000
        
        speedup = pytorch_time / cuda_time
        
        print(f"  PyTorch: {pytorch_time:.2f} ms")
        print(f"  CUDA:    {cuda_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")

def test_different_dtypes():
    """Test kernel with different data types"""
    print("\n\nTesting different dtypes...")
    
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    for dtype in dtypes:
        print(f"\nTesting {dtype}:")
        
        G = torch.randn(128, 128, dtype=dtype, device='cuda')
        
        try:
            result = zeropower_via_newtonschulz5(G, steps=5)
            print(f"  Success! Output shape: {result.shape}, dtype: {result.dtype}")
            
            # Check orthogonality
            orth_error = torch.norm(result @ result.T - torch.eye(128, device='cuda', dtype=dtype)).item()
            print(f"  Orthogonality error: {orth_error:.6f}")
        except Exception as e:
            print(f"  Failed with error: {e}")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n\nTesting edge cases...")
    
    # Very small matrix
    G = torch.randn(2, 2, device='cuda')
    result = zeropower_via_newtonschulz5(G, steps=5)
    print(f"Very small matrix (2x2): Success")
    
    # Very large aspect ratio
    G = torch.randn(8, 1024, device='cuda')
    result = zeropower_via_newtonschulz5(G, steps=5)
    print(f"Large aspect ratio (8x1024): Success")
    
    # Single precision limits
    G = torch.randn(64, 64, device='cuda') * 1e10
    result = zeropower_via_newtonschulz5(G, steps=5)
    print(f"Large values: Success")

if __name__ == "__main__":
    print("Newton-Schulz CUDA Kernel Test Suite")
    print("=" * 50)
    
    test_correctness()
    benchmark_performance()
    test_different_dtypes()
    test_edge_cases()
    
    print("\n\nAll tests completed!")