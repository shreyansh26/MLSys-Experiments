import torch
import triton
import triton.language as tl
from triton import Config
from triton_util import get_1d_offset, get_2d_offset, get_1d_mask, get_2d_mask
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

_ordered_datatypes = [torch.int8, torch.float16, torch.bfloat16, torch.float32]

def get_higher_dtype(dtype_a, dtype_b):
    if dtype_a is dtype_b:
        return dtype_a

    assert dtype_a in _ordered_datatypes
    assert dtype_b in _ordered_datatypes

    for d in _ordered_datatypes:
        if dtype_a is d:
            return dtype_b
        if dtype_b is d:
            return dtype_a

@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def matmul_kernel(
        A, B, C, 
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        acc_dtype: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        AB_DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m_t, pid_n_t = pid // num_pid_n, pid % num_pid_n 

    pid_m, pid_n = tl.swizzle2d(pid_m_t, pid_n_t, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_m = get_1d_offset(BLOCK_M, pid_m)
    offs_n = get_1d_offset(BLOCK_N, pid_n)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
    offs_k = get_1d_offset(BLOCK_K, pid_z)

    offs_amk = get_2d_offset(offs_am, offs_k, stride_0=stride_am, stride_1=stride_ak)
    offs_bkn = get_2d_offset(offs_k, offs_bn, stride_0=stride_bk, stride_1=stride_bn)

    A = A + offs_amk
    B = B + offs_bkn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=offs_k[:, None] < k_remaining, other=_0)
            b = tl.load(B, mask=offs_k[None, :] < k_remaining, other=_0)

        print("ar shape", a.shape[0])
        print("ac shape", a.shape[1])

        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)

        acc += tl.dot(a, b, out_dtype=acc_dtype)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    acc = acc.to(C.type.element_ty)

    # rematerialize offs_m and offs_n to save registers
    offs_m = get_1d_offset(BLOCK_M, pid_m)
    offs_n = get_1d_offset(BLOCK_N, pid_n)

    offs_cmn = get_2d_offset(offs_m, offs_n, stride_cm, stride_cn)
    
    C = C + offs_cmn
    mask = get_2d_mask(offs_m, offs_n, M, N)

    # write back with reduction splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)

def matmul(a, b, acc_dtype=None, output_dtype=None):
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    M, K = a.shape
    K, N = b.shape

    ab_dtype = get_higher_dtype(a.dtype, b.dtype)

    if output_dtype is None:
        output_dtype = ab_dtype

    c = torch.empty((M, N), device=a.device, dtype=output_dtype)

    # Allowed types for acc_type given the types of a and b.
    supported_acc_dtypes = {
        torch.float16: (torch.float32, torch.float16), 
        torch.bfloat16: (torch.float32, torch.bfloat16),
        torch.float32: (torch.float32, ),
        torch.int8: (torch.int32, )
    }

    if acc_dtype is None:
        acc_dtype = supported_acc_dtypes[ab_dtype][0]
    else:
        assert isinstance(acc_dtype, torch.dtype), "acc_dtype must be a torch.dtype"
        assert acc_dtype in supported_acc_dtypes[a.dtype], "acc_dtype not compatible with the type of a"
        assert acc_dtype in supported_acc_dtypes[b.dtype], "acc_dtype not compatible with the type of b"

    def to_tl_type(ty):
        return getattr(tl, str(ty).split(".")[-1])

    acc_dtype = to_tl_type(acc_dtype)
    ab_dtype = to_tl_type(ab_dtype)
    output_dtype = to_tl_type(output_dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        acc_dtype=acc_dtype,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=16,
        SPLIT_K=1,
        GROUP_SIZE_M=1,
        AB_DTYPE=ab_dtype
    )
    return c

def test():
    a = torch.randn((1024, 1024), device='cuda', dtype=torch.float32)
    b = torch.randn((1024, 1024), device='cuda', dtype=torch.float32)

    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)

    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")

    if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=1e-1):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

def calc_flops(M, K, N):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    ms = triton.testing.do_bench(lambda: matmul(a, b))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms)

def run(M, K, N):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = matmul(a, b)
    return c

M = 1024
K = N = 1024

for i in range(10):
    c = run(M, K, N)

c = run(M, K, N)
print(c.shape)