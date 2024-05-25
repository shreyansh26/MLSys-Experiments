import torch
import triton
import triton.language as tl
from triton import Config
from triton_util import get_1d_offset, get_2d_offset, get_1d_mask, get_2d_mask
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

_ordered_datatypes = [torch.int8, torch.float16, torch.bfloat16, torch.float32]

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

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

def get_configs_io_bound(do_split_k=False, do_col_major=False):
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    
                    if do_split_k:
                        for split_k in [2, 4, 8]:
                            configs.append(
                                Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k, 'GROUP_SIZE_M': 8},
                                    num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
                    elif do_col_major:
                        configs.append(
                        Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                               num_stages=num_stages, num_warps=num_warps))
                    else:
                        configs.append(
                        Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8},
                               num_stages=num_stages, num_warps=num_warps))
    return configs                    

@triton.jit()
def col_major(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    pid_m = (pid % grid_m)
    pid_n = pid // grid_m

    return pid_m, pid_n

@triton.autotune(
    configs=get_configs_io_bound(do_split_k=True),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def matmul_kernel_grouped_splitk(
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

@triton.autotune(
    configs=get_configs_io_bound(do_split_k=False),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def matmul_kernel_grouped(
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

@triton.autotune(
    configs=get_configs_io_bound(do_col_major=True),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def matmul_kernel_col_major(
        A, B, C, 
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        acc_dtype: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        AB_DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    pid_m, pid_n = col_major(pid, M, N, BLOCK_M, BLOCK_N)

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

def matmul(a, b, kernel_name, acc_dtype=None, output_dtype=None):
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
    
    kernel_name[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        acc_dtype=acc_dtype,
        AB_DTYPE=ab_dtype
    )
    return c

def test(kernel_name):
    a = torch.randn((10, 1024), device='cuda', dtype=torch.float32)
    b = torch.randn((1024, 1024), device='cuda', dtype=torch.float32)

    triton_output = matmul(a, b, kernel_name=kernel_name)
    torch_output = torch.matmul(a, b)

    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")

    if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=1e-1):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

def calc_flops(M, K, N, kernel_name):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    ms = triton.testing.do_bench(lambda: matmul(a, b, kernel_name))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms)

def calc_flops_triton(M, K, N):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    ms = triton.testing.do_bench(lambda: triton.ops.matmul(a, b))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms)

test(matmul_kernel_grouped)
test(matmul_kernel_grouped_splitk)
test(matmul_kernel_col_major)

configs = []
for m_val in [1, 2, 4, 8, 16]:
    configs.append(triton.testing.Benchmark(
        x_names=["N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(9, 15)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=["matmul_kernel_grouped", "matmul_kernel_grouped_splitk", "matmul_kernel_col_major"],  # Label name for the lines
        line_names=["matmul_kernel_grouped", "matmul_kernel_grouped_splitk", "matmul_kernel_col_major"],  # Line styles
        styles=[("green", "-"), ("blue", "-"), ("orange", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name=f"matmul-performance-fp16-m{m_val}",
        args={"M": m_val},
    ))

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "matmul_kernel_grouped":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, matmul_kernel_grouped), quantiles=quantiles)
    if provider == "matmul_kernel_grouped_splitk":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, matmul_kernel_grouped_splitk), quantiles=quantiles)
    if provider == "matmul_kernel_col_major":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, matmul_kernel_col_major), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True, save_path="plots")