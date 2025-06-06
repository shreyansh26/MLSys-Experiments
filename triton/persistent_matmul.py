import torch

import triton
import triton.language as tl
import triton.profiler as proton

from contextlib import contextmanager

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 10

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret

HAS_TENSOR_DESC = supports_tma() and hasattr(tl, "make_tensor_descriptor")
HAS_HOST_TENSOR_DESC = supports_tma() and hasattr(triton.tools.tensor_descriptor, "TensorDescriptor")
HAS_WARP_SPECIALIZE = supports_ws() and HAS_TENSOR_DESC

print(f"HAS_TENSOR_DESC: {HAS_TENSOR_DESC}")
print(f"HAS_HOST_TENSOR_DESC: {HAS_HOST_TENSOR_DESC}")
print(f"HAS_WARP_SPECIALIZE: {HAS_WARP_SPECIALIZE}")

@triton.jit
def get_group_pids(pid, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n

def matmul_get_configs(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8}, num_stages=s, num_warps=w, pre_hook=pre_hook) \
        for BM in [64, 128, 256] \
        for BN in [64, 128, 256] \
        for BK in [64, 128, 256] \
        for s in ([3, 4, 5]) \
        for w in [4, 8] \
    ]

@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_naive_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_m, pid_n = get_group_pids(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offset_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offset_bn = start_n + tl.arange(0, BLOCK_SIZE_N)

    offset_am = tl.where(offset_am < M, offset_am, 0)
    offset_bn = tl.where(offset_bn < N, offset_bn, 0)

    offset_am = tl.max_contiguous(tl.multiple_of(offset_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offset_bn = tl.max_contiguous(tl.multiple_of(offset_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptrs = b_ptr + offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offset_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offset_k[:, None] < K - k, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        a_ptrs = a_ptrs + stride_ak * BLOCK_SIZE_K
        b_ptrs = b_ptrs + stride_bk * BLOCK_SIZE_K

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offset_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offset_cm = tl.where(offset_cm < M, offset_cm, 0)
    offset_cn = tl.where(offset_cn < N, offset_cn, 0)
    c_ptrs = c_ptr + offset_cm[:, None] * stride_cm + offset_cn[None, :] * stride_cn
    c_mask = (offset_cm[:, None] < M) & (offset_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_naive_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def cublas_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"cublas [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        cublas.matmul(a, b, c)
    return c

def torch_matmul(a, b):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"torch [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        c = torch.matmul(a, b.T)
    return c

def custom_allclose(a, b, rtol=1e-2, atol=1e-2):
    # float8 matmuls are not super precise
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    return torch.allclose(a, b, rtol=rtol, atol=atol)

@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)

def bench_fn(label, reps, warmup_reps, fn, *args):
    print(f"Benchmarking {label}: ...", end="")
    for _ in range(warmup_reps):
        fn(*args)
    with proton_context():
        for _ in range(reps):
            fn(*args)
    print(f"\rBenchmarking {label}: done")

def bench(K, dtype, reps=10000, warmup_reps=10000):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    b = b.T.contiguous()

    if cublas is not None:
        bench_fn("cublas", reps, warmup_reps, cublas_matmul, a, b)
    if dtype == torch.float16:
        bench_fn("torch", reps, warmup_reps, torch_matmul, a, b)
    bench_fn("naive", reps, warmup_reps, matmul, a, b.T)

if __name__ == "__main__":
    FP8 = True
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    if FP8:
        a = a.to(torch.float8_e4m3fn)
        b = b.to(torch.float8_e4m3fn)

    b_t = b.T.contiguous()
    c_naive_triton = matmul(a, b)
    c_cublas = cublas_matmul(a, b_t)

    if not FP8:
        c_torch = torch_matmul(a, b_t)

    if custom_allclose(c_naive_triton, c_cublas):
        print(f"✅ Naive Triton and Cublas match. dtype {a.dtype}")
    else:
        print(f"❌ Naive Triton and Cublas do not match. dtype {a.dtype}")
    if not FP8:
        if custom_allclose(c_naive_triton, c_torch):
            print(f"✅ Naive Triton and Torch match. dtype {a.dtype}")
        else:
            print(f"❌ Naive Triton and Torch do not match. dtype {a.dtype}")

    proton.start("matmul_fp8" if FP8 else "matmul_fp16", hook="triton")
    proton.deactivate()
    for K in range(1024, 8192 + 1, 1024):
        bench(K, a.dtype)
    proton.finalize()