import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice

from pathlib import Path

DEVICE = torch.device('cuda')
torch.manual_seed(0)

@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)

if __name__ == '__main__':
    size = 98432
    x = torch.rand(size, device=DEVICE)

    output_triton = torch.zeros(size, device=DEVICE)
    output_torch = torch.asin(x)
    assert x.is_cuda and output_triton.is_cuda

    n_elements = output_torch.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # Using default libdevice
    asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
    
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')

    # Custom path to libdevice
    libdir = Path(__file__).parent / 'third_party/nvidia/backend/lib'
    extern_libs = {'libdevice': str(libdir / 'libdevice.10.bc')}

    output_triton = torch.empty_like(x)
    asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024, extern_libs=extern_libs)
    
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')