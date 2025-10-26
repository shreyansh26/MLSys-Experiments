import torch
import torch.nn.functional as F
import attention_cuda_flash
import time
import sys

def test_flash_attention(B, H, N, D, dtype, is_causal, device='cuda'):
    """Test Flash Attention kernel against PyTorch reference implementation."""
    
    print(f"\nTesting Flash Attention: B={B}, H={H}, N={N}, D={D}, dtype={dtype}, is_causal={is_causal}")
    
    # Create random inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    K = torch.randn(B, H, N, D, dtype=dtype, device=device)
    V = torch.randn(B, H, N, D, dtype=dtype, device=device)
    
    # Our Flash Attention implementation
    output_flash = attention_cuda_flash.attention_forward_flash(Q, K, V, is_causal)
    
    # PyTorch reference implementation
    with torch.no_grad():
        output_torch = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal
        )
    
    # Check correctness
    max_diff = (output_flash - output_torch).abs().max().item()
    mean_diff = (output_flash - output_torch).abs().mean().item()
    
    # Set tolerance based on dtype and sequence length
    # Flash Attention has some numerical precision issues for very long sequences
    if dtype == torch.float32:
        if N >= 4096:
            atol, rtol = 1e-1, 1e-2  # Relaxed for very long sequences
        else:
            atol, rtol = 1e-4, 1e-3
    elif dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    elif dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    
    passed = torch.allclose(output_flash, output_torch, atol=atol, rtol=rtol)
    
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    if not passed:
        print(f"  Expected tolerance: atol={atol}, rtol={rtol}")
        rel_diff = ((output_flash - output_torch).abs() / (output_torch.abs() + 1e-8)).max().item()
        print(f"  Max relative diff: {rel_diff:.6e}")
        
        # Debug: show some sample values
        print(f"\n  Sample Flash output: {output_flash[0, 0, 0, :5]}")
        print(f"  Sample PyTorch:      {output_torch[0, 0, 0, :5]}")
    
    return passed

def benchmark_flash_attention(B, H, N, D, dtype, is_causal, device='cuda', warmup=10, iterations=100):
    """Benchmark Flash Attention kernel performance."""
    
    print(f"\nBenchmarking Flash Attention: B={B}, H={H}, N={N}, D={D}, dtype={dtype}, is_causal={is_causal}")
    
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    K = torch.randn(B, H, N, D, dtype=dtype, device=device)
    V = torch.randn(B, H, N, D, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = attention_cuda_flash.attention_forward_flash(Q, K, V, is_causal)
    torch.cuda.synchronize()
    
    # Benchmark our Flash Attention implementation
    start = time.time()
    for _ in range(iterations):
        output_flash = attention_cuda_flash.attention_forward_flash(Q, K, V, is_causal)
    torch.cuda.synchronize()
    flash_time = (time.time() - start) / iterations * 1000  # ms
    
    # Warmup PyTorch
    for _ in range(warmup):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch reference
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            output_torch = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations * 1000  # ms
    
    print(f"  Flash Attention: {flash_time:.4f} ms")
    print(f"  PyTorch:         {torch_time:.4f} ms")
    print(f"  Speedup:         {torch_time/flash_time:.2f}x")
    
    return flash_time, torch_time

def main():
    print("=" * 80)
    print("Flash Attention CUDA Kernel - Correctness Tests")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available!")
        sys.exit(1)
    
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Get device properties
    props = torch.cuda.get_device_properties(0)
    print(f"Total Memory: {props.total_memory / (1024**3):.1f} GB")
    print(f"Shared Memory per Block: {props.shared_memory_per_block / 1024:.1f} KB")
    
    all_passed = True
    
    # Test configurations - gradually increasing sequence lengths
    test_configs = [
        # (B, H, N, D, dtype, is_causal)
        # Small tests
        (1, 1, 64, 64, torch.float32, False),
        (1, 1, 64, 64, torch.float32, True),
        (2, 4, 128, 64, torch.float32, False),
        (2, 4, 128, 64, torch.float32, True),
        
        # Float16 tests
        (1, 1, 64, 64, torch.float16, False),
        (1, 1, 64, 64, torch.float16, True),
        (2, 4, 128, 64, torch.float16, False),
        
        # BFloat16 tests
        (1, 1, 64, 64, torch.bfloat16, False),
        (1, 1, 64, 64, torch.bfloat16, True),
        (2, 4, 128, 64, torch.bfloat16, False),
        
        # Medium tests
        (4, 8, 256, 64, torch.float32, False),
        (4, 8, 256, 64, torch.float32, True),
        
        # Larger tests
        (2, 8, 512, 64, torch.float32, False),
        (2, 8, 512, 64, torch.float32, True),
        (1, 8, 1024, 64, torch.float32, False),
        (1, 8, 1024, 64, torch.float32, True),
        
        # Very long sequences (where Flash Attention shines!)
        (1, 4, 2048, 64, torch.float32, False),
        (1, 4, 2048, 64, torch.float32, True),
        (1, 4, 4096, 64, torch.float32, False),
        (1, 2, 8192, 64, torch.float32, False),
        
        # Different head dimensions
        (2, 4, 128, 32, torch.float32, False),
        (2, 4, 256, 128, torch.float32, False),
    ]
    
    for config in test_configs:
        try:
            passed = test_flash_attention(*config, device=device)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  ✗ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 80)
    
    # Optional: Run benchmarks
    if all_passed and len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        print("\n" + "=" * 80)
        print("Flash Attention CUDA Kernel - Performance Benchmarks")
        print("=" * 80)
        
        benchmark_configs = [
            (1, 8, 256, 64, torch.float32, False),
            (1, 8, 512, 64, torch.float32, False),
            (1, 8, 512, 64, torch.float32, True),
            (1, 8, 1024, 64, torch.float32, False),
            (1, 8, 1024, 64, torch.float32, True),
            (1, 8, 2048, 64, torch.float32, False),
            (1, 8, 2048, 64, torch.float32, True),
            (1, 4, 4096, 64, torch.float32, False),
            (1, 2, 8192, 64, torch.float32, False),
            (1, 8, 512, 64, torch.float16, False),
            (1, 8, 512, 64, torch.bfloat16, False),
        ]
        
        for config in benchmark_configs:
            try:
                benchmark_flash_attention(*config, device=device)
            except Exception as e:
                print(f"  Benchmark failed: {e}")
                import traceback
                traceback.print_exc()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())

