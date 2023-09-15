# Vector addition
import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_ele,
    BLOCK_SIZE: tl.constexpr # Otherwise, ValueError: arange's arguments must be of type tl.constexpr
):
    # Get PID
    pid = tl.program_id(axis=0) # axis = 0 for 1D grid

    # Get offsets for processing
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # Create mask to prevent out-of-bound accesses
    mask = offsets < n_ele

    # Load data for computation from DRAM and mask out eextra elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    out = x + y

    # Write output to DRAM
    tl.store(out_ptr + offsets, out, mask)

def add(x: torch.tensor, y: torch.tensor):
    # Preallocate output
    out = torch.empty_like(x)

    assert x.is_cuda and y.is_cuda and out.is_cuda

    n_ele = out.numel()

    # SPMD launch grid denotes number of kernels that run in parallel
    # Similar to CUDA launch grid.
    # Can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks.
    grid = lambda meta: (triton.cdiv(n_ele, meta['BLOCK_SIZE']), )

    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, out, n_ele, BLOCK_SIZE=1024)

    return out

torch.manual_seed(100)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(
    f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}'
)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        xlabel='Size',
        x_vals=[
            2 ** i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, percentiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), percentiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6 # FP32 = 4 bytes. 2 Read 1 Write = 3. => 4 Bytes * 3 Memory Accesses * Num Elements / Time
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, save_path='plots/01-vector-add')
