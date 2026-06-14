import torch
import triton

def decomposeK(a, b, k_splits):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    assert k == b.shape[0], "Incompatible dimensions"
    assert k % k_splits == 0, "k must be divisible by k_splits"

    k_parts = k // k_splits

    a_reshaped = a.reshape(m, k_splits, k_parts).permute(1, 0, 2) # [m, k_splits, k_parts] -> [k_splits, m, k_parts]
    b_reshaped = b.reshape(k_splits, k_parts, n) # [k_splits, k_parts, n]

    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    reduced_result = result.sum(dim=0)
    return reduced_result.to(a.dtype)

def torch_matmul(a, b):
    return torch.matmul(a, b)

def bench(label, fn, flops):
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"{label}: {ms:.3f} ms, {tflops:.2f} TFLOP/s")

if __name__ == "__main__":
    dtype = torch.bfloat16
    device = "cuda"
    a = torch.randn(64, 7168, dtype=dtype, device=device)
    b = torch.randn(7168, 256, dtype=dtype, device=device)

    decomposeK_compiled = torch.compile(decomposeK, mode="max-autotune-no-cudagraphs")
    result_decomposeK = decomposeK(a, b, 4)
    result_decomposeK_compiled = decomposeK_compiled(a, b, 4)
    result_torch_matmul = torch_matmul(a, b)

    print("result_decomposeK:")
    print(result_decomposeK.shape)
    print("--------------------------------")
    print("result_decomposeK_compiled:")
    print(result_decomposeK_compiled.shape)
    print("--------------------------------")
    print("result_torch_matmul:")
    print(result_torch_matmul.shape)
    print("--------------------------------")
    torch.testing.assert_close(result_decomposeK, result_torch_matmul, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(result_decomposeK_compiled, result_torch_matmul, rtol=2e-2, atol=1e-2)

    flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
    bench("decomposeK", lambda: decomposeK(a, b, 4), flops)
    bench("decomposeK_compiled", lambda: decomposeK_compiled(a, b, 4), flops)
    bench("torch.matmul", lambda: torch_matmul(a, b), flops)