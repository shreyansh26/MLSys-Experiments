import torch

import triton
import triton.language as tl

from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': num_sms(),
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 64,
            'NUM_SM': num_sms(),
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    group_a_ptrs,               # shape: [group_size], each entry is a pointer to to a group of A (A_i)
    group_b_ptrs,               # shape: [group_size], each entry is a pointer to to a group of B (B_i)
    group_c_ptrs,               # shape: [group_size], each entry is a pointer to to a group of C (C_i)
    group_gemm_sizes,           # shape: [group_size * 3], each entry is the size of the gemm operation (M_i, N_i, K_i) for each i (group_i)
    group_strides,              # shape: [group_size * 3], each entry is the stride of the gemm operation (stride_a_i, stride_b_i, stride_c_i) for each i (group_i)
    group_size,                 # number of gemm operations
    NUM_SM: tl.constexpr,       # number of sms to use (CTAs to launch)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0            # to keep track of cumulative tile counts of all previous GEMMs
                                    # this helps to figure out which GEMM the current tile belongs to

    for g in range(group_size):
        # get gemm size for group g
        gm = tl.load(group_gemm_sizes + g * 3 + 0)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)

        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        # iterate through tiles of current GEMM (group g)
        while(tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile for current GEMM (group g)
            stride_a = tl.load(group_strides + g * 3 + 0)
            stride_b = tl.load(group_strides + g * 3 + 1)
            stride_c = tl.load(group_strides + g * 3 + 2)

            # get pointers for group g
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

            # tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            m_tile_idx = tile_idx_in_gemm // num_n_tiles
            n_tile_idx = tile_idx_in_gemm % num_n_tiles

            # compute GEMM
            offset_am = m_tile_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offset_bn = n_tile_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offset_k = tl.arange(0, BLOCK_SIZE_K)

            # pointer calculation for start of each tile
            a_ptrs = a_ptr + offset_am[:, None] * stride_a + offset_k[None, :]
            b_ptrs = b_ptr + offset_k[:, None] * stride_b + offset_bn[None, :]

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for kk in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
                # hint for Triton to do loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])

                # Calculate the current K offset for masking
                k_offset = kk * BLOCK_SIZE_K

                # load tile
                a = tl.load(a_ptrs, mask=(offset_am[:, None] < gm) & (k_offset + offset_k[None, :] < gk), other=0.0)
                b = tl.load(b_ptrs, mask=(k_offset + offset_k[:, None] < gk) & (offset_bn[None, :] < gn), other=0.0)

                # no masking is faster but incorrect for non-block-size-aligned tiles
                # a = tl.load(a_ptrs)
                # b = tl.load(b_ptrs)

                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * stride_b

            c = accumulator.to(tl.float16)

            offset_cm = m_tile_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offset_cn = n_tile_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            c_ptrs = c_ptr + offset_cm[:, None] * stride_c + offset_cn[None, :]

            # store the result
            tl.store(c_ptrs, c, mask=(offset_cm[:, None] < gm) & (offset_cn[None, :] < gn))
            
            # no masking is faster but incorrect for non-block-size-aligned tiles
            # tl.store(c_ptrs, c)

            # go to the next tile - somewhat like grid-stride loop
            tile_idx += NUM_SM

        # go to the next GEMM
        last_problem_end += num_tiles

tma_configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K' : BK}, num_stages=s, num_warps=w) \
    for BM in [128]\
    for BN in [128, 256]\
    for BK in [64, 128]\
    for s in ([3, 4])\
    for w in [4, 8]\
]


@triton.autotune(
    tma_configs,
    key=['group_a_ptrs', 'group_b_ptrs', 'gropup_c_ptrs', 'group_size'],
)
@triton.jit
def grouped_matmul_tma_kernel(
    group_a_ptrs,               # shape: [group_size], each entry is a pointer to to a group of A (A_i)
    group_b_ptrs,               # shape: [group_size], each entry is a pointer to to a group of B (B_i)
    group_c_ptrs,               # shape: [group_size], each entry is a pointer to to a group of C (C_i)
    group_gemm_sizes,           # shape: [group_size * 3], each entry is the size of the gemm operation (M_i, N_i, K_i) for each i (group_i)
    group_strides,              # shape: [group_size * 3], each entry is the stride of the gemm operation (stride_a_i, stride_b_i, stride_c_i) for each i (group_i)
    group_size,                 # number of gemm operations
    NUM_SM: tl.constexpr,       # number of sms to use (CTAs to launch)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    FP8: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8 else tl.float16
    tile_idx = tl.program_id(0)
    last_problem_end = 0

    for g in range(group_size):
        # get gemm size for group g
        gm = tl.load(group_gemm_sizes + g * 3 + 0)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)

        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile for current GEMM (group g)
            stride_a = tl.load(group_strides + g * 3 + 0)
            stride_b = tl.load(group_strides + g * 3 + 1)
            stride_c = tl.load(group_strides + g * 3 + 2)

            # get pointers for group g
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))

            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[stride_a, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )

            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[stride_b, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[stride_c, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                # tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                m_tile_idx = tile_idx_in_gemm // num_n_tiles
                n_tile_idx = tile_idx_in_gemm % num_n_tiles

                # compute GEMM
                offset_am = m_tile_idx * BLOCK_SIZE_M
                offset_bn = n_tile_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                for kk in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
                    a = a_desc.load([offset_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offset_bn, kk * BLOCK_SIZE_K])
                    accumulator += tl.dot(a, b.T)
                
                offset_cm = m_tile_idx * BLOCK_SIZE_M
                offset_cn = n_tile_idx * BLOCK_SIZE_N

                c = accumulator.to(dtype)
                c_desc.store([offset_cm, offset_cn], c)

                # go to the next tile - somewhat like grid-stride loop
                tile_idx += NUM_SM

        last_problem_end += num_tiles
        
def group_gemm_fn(group_A, group_B, use_tma=False):
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_strides = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]

        if use_tma:
            assert A.shape[1] == B.shape[1]
            M, K = A.shape
            N, K = B.shape
        else:
            assert A.shape[1] == B.shape[0]
            M, K = A.shape
            K, N = B.shape
        
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_strides += [A.stride(0), B.stride(0), C.stride(0)]

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_strides = torch.tensor(g_strides, dtype=torch.int32, device=DEVICE)

    if use_tma:
        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)
    
        triton.set_allocator(alloc_fn)

        # we use a fixed number of CTA, and it's auto-tunable
        grid = lambda META: (META['NUM_SM'], )
        grouped_matmul_tma_kernel[grid](
            d_a_ptrs, 
            d_b_ptrs, 
            d_c_ptrs, 
            d_g_sizes, 
            d_g_strides, 
            group_size, 
            FP8=torch.float8_e4m3fn == group_A[0].dtype, 
            NUM_SM=num_sms())
    else:
        # we use a fixed number of CTA, and it's auto-tunable
        grid = lambda META: (META['NUM_SM'], )
        grouped_matmul_kernel[grid](
            d_a_ptrs,
            d_b_ptrs,
            d_c_ptrs,
            d_g_sizes,
            d_g_strides,
            group_size,
        )

    # Since we are modifying the data pointers of each C, group_C now has the updated C tensors
    return group_C

def create_perf_report(mode, bench_type, x_name="N", y_label="GB/s"):
    plot_name = f"grouped-gemm-{mode}-{bench_type}"
    return triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[x_name],
            x_vals=[2**i for i in range(7, 11)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel=y_label,
            plot_name=plot_name,
            args={'mode': mode},
        ))

def bench_group_gemm_generic(M, N, K, provider="triton", bench_type="latency"):
    group_size = 4
    group_A = []
    group_B = []
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
        C = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
        group_A.append(A)
        group_B.append(B)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    quantiles = [0.5, 0.2, 0.8]

    def torch_perf_fn(group_A, group_B):
        for a, b in zip(group_A, group_B):
            torch.matmul(a, b)
    
    def triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size):
        grid = lambda META: (META['NUM_SM'], )
        grouped_matmul_kernel[grid](
            d_a_ptrs,
            d_b_ptrs,
            d_c_ptrs,
            d_g_sizes,
            d_g_lds,
            group_size,
        )

    def gbps(ms):
        flops = 0
        for i in range(group_size):
            flops += 2 * g_sizes[i*3] * g_sizes[i*3+1] * g_sizes[i*3+2]
        return flops * 1e-9 / ms

    if bench_type == "latency":
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size), quantiles=quantiles)
        return ms, min_ms, max_ms
    
    elif bench_type == "flops":
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size), quantiles=quantiles)
        return gbps(ms), gbps(max_ms), gbps(min_ms)

def make_benchmark(bench_type, mode, x_name="N", y_label="GB/s", M=None, N=None, K=None):
    @create_perf_report(mode, bench_type, x_name, y_label)
    def bench(provider, **kwargs):
        if mode == "square":
            M = kwargs['N']
            N = kwargs['N']
            K = kwargs['N']
        elif mode == "batch":
            M = kwargs['M']
            N = 8192
            K = 8192
            
        return bench_group_gemm_generic(M=M, N=N, K=K, provider=provider, bench_type=bench_type)
    return bench

if __name__ == "__main__":
    # Data preparation
    # TODO: Add back hard case of 1024 + 13 to work with TMA
    group_m = [1024, 512, 256, 128]
    group_n = [1024, 512, 256, 128]
    group_k = [1024, 512, 256, 128]
    group_A = []
    group_B = []
    group_B_T = []
    assert len(group_m) == len(group_n)
    assert len(group_n) == len(group_k)
    group_size = len(group_m)
    for i in range(group_size):
        M = group_m[i]
        N = group_n[i]
        K = group_k[i]
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)

    # Calculate the output of the group gemm using Triton
    out_triton = group_gemm_fn(group_A, group_B)        
    # Calculate the output of the group gemm using torch
    out_ref = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
    # Check if the output of the group gemm using Triton is the same as the output of the group gemm using torch
    for i in range(group_size):
        assert torch.allclose(out_ref[i], out_triton[i], atol=1e-2, rtol=1e-2)
        print(f"✅ Group {i} passed")

    if supports_tma():
        # Calculate the output of the group gemm using Triton with TMA
        out_triton_tma = group_gemm_fn(group_A, group_B_T, use_tma=True)
        for i in range(group_size):
            assert torch.allclose(out_ref[i], out_triton_tma[i], atol=1e-2, rtol=1e-2)
            print(f"✅ Group {i} (TMA) passed")

    # bench_grouped_gemm_flops_square = make_benchmark("flops", "square", x_name="N", y_label="GBs/s")
    # bench_grouped_gemm_flops_batch = make_benchmark("flops", "batch", x_name="M", y_label="GBs/s")
    # bench_grouped_gemm_latency_square = make_benchmark("latency", "square", x_name="N", y_label="ms")
    # bench_grouped_gemm_latency_batch = make_benchmark("latency", "batch", x_name="M", y_label="ms")

    # bench_grouped_gemm_flops_square.run(save_path='plots/grouped_gemm', print_data=True)
    # bench_grouped_gemm_flops_batch.run(save_path='plots/grouped_gemm', print_data=True)
    # bench_grouped_gemm_latency_square.run(save_path='plots/grouped_gemm', print_data=True)
    # bench_grouped_gemm_latency_batch.run(save_path='plots/grouped_gemm', print_data=True)