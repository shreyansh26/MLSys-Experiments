# Fused Softmax
import torch

import triton
import triton.language as tl

'''
When implemented naively in PyTorch, computing y = naive_softmax(x) for 
 requires reading 
 elements from DRAM and writing back 
 elements. This is obviously wasteful; we'd prefer to have a custom “fused” kernel that only reads X once and does all the necessary computations on-chip. Doing so would require reading and writing back only 
 bytes, so we could expect a theoretical speed-up of ~4x (i.e., 
). The torch.jit.script flags aims to perform this kind of “kernel fusion” automatically but, as we will see later, it is still far from ideal.
'''
@torch.jit.script
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

'''
X - m x n matrix
Assume row-wise softmax (dim=1 in Pytorch)
Have to calculate sum (e^x) of each row

Each thread operates on one row.

Block size is entire column length (just largest power of 2)
'''
@triton.jit
def fused_softmax_kernel(
    x_ptr,
    out_ptr,
    n_col,
    x_row_stride,
    out_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)

    x_row_start = x_ptr + row_id * x_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    x_row_ptrs = x_row_start + col_offsets

    mask = col_offsets < n_col

    # Load into SRAM, mask values and make others -inf
    x_row = tl.load(x_row_ptrs, mask=mask, other=-float('inf'))

    x_row_minus_max = x_row - tl.max(x_row, axis=0)
    # Exponentioation in Triton is fast but approximate
    x_row_minus_max_exp = tl.exp(x_row_minus_max)
    x_row_minus_max_exp_div_sum = x_row_minus_max_exp / tl.sum(x_row_minus_max_exp, axis=0)
    
    out_row_start = out_ptr + row_id * out_row_stride
    out_row_ptrs = out_row_start + col_offsets

    # Save into DRAM
    tl.store(out_row_ptrs, x_row_minus_max_exp_div_sum, mask=mask)

def fused_softmax(x: torch.tensor):
    rows, cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    out = torch.empty_like(x)

    fused_softmax_kernel[(rows,)](x, out, cols, x.stride(0), out.stride(0), num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)

    return out

torch.manual_seed(0)
x = torch.randn(16384, 768, device='cuda')
y_triton = fused_softmax(x)
y_torch = torch.softmax(x, axis=1)

assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],  # Argument names to use as an x-axis for the plot.
        xlabel='M',
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # Different possible values for `x_name`.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch-jit', 'torch-native'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch (JIT)', 'Torch (Native)'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='fused-softmax-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={'N': 4096},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(N, M, provider):
    print(N, M)
    x = torch.rand(N, M, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1), percentiles=quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), percentiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_softmax(x), percentiles=quantiles)
    gbps = lambda ms: (2 * x.numel() * x.element_size() * 1e-9) / (ms * 1e-3) # 2MN memory access - MN read + MN write
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, save_path='plots/02-fused-softmax')