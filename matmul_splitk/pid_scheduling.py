import torch
import triton
import triton.language as tl
import numpy as np
from triton_util import get_1d_offset, get_2d_offset, get_2d_mask

TOTAL_PROGRAMS = 16
M = 8
N = 8
GROUP_SIZE_M = 2
BLOCK_SIZE_M = 2
BLOCK_SIZE_N = 2
BLOCKS_M = triton.cdiv(M, BLOCK_SIZE_M)
BLOCKS_N = triton.cdiv(N, BLOCK_SIZE_N)

x = torch.arange(BLOCKS_M * BLOCKS_N, device='cuda').view(BLOCKS_M, BLOCKS_N)
z = torch.ones_like(x) * -1

@triton.jit()
def grouped_launch(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n

@triton.jit()
def col_major(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    pid_m = (pid % grid_n)
    pid_n = pid // grid_m

    return pid_m, pid_n

@triton.jit
def grouped(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n 

    pid_m_, pid_n_ = grouped_launch(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    offs_m = get_1d_offset(1, n_prev_chunks=pid_m)
    offs_n = get_1d_offset(1, n_prev_chunks=pid_n)
    
    offs = get_2d_offset(offs_m, offs_n, stride_0=num_pid_n)
    mask = get_2d_mask(offs_m, offs_n, max_0=num_pid_m, max_1=num_pid_n)

    offs_sw_m = get_1d_offset(1, n_prev_chunks=pid_m_)
    offs_sw_n = get_1d_offset(1, n_prev_chunks=pid_n_)
    
    offs_sw = get_2d_offset(offs_sw_m, offs_sw_n, stride_0=num_pid_n)
    mask_sw = get_2d_mask(offs_sw_m, offs_sw_n, max_0=num_pid_m, max_1=num_pid_n)
    
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(z_ptr + offs_sw, x, mask=mask_sw)

@triton.jit
def swizzle_k_2d(x_ptr, z_ptr, GROUP_SIZE_M: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    num_pid_m, num_pid_n = tl.num_programs(0), tl.num_programs(1)

    pid_m_, pid_n_ = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)  # Weirdness: tl.swizzle2d doesn't work when simulating on CPU
    
    offs_m = get_1d_offset(1, n_prev_chunks=pid_m)
    offs_n = get_1d_offset(1, n_prev_chunks=pid_n)
    
    offs = get_2d_offset(offs_m, offs_n, stride_0=num_pid_n)
    mask = get_2d_mask(offs_m, offs_n, max_0=num_pid_m, max_1=num_pid_n)

    offs_sw_m = get_1d_offset(1, n_prev_chunks=pid_m_)
    offs_sw_n = get_1d_offset(1, n_prev_chunks=pid_n_)
    
    offs_sw = get_2d_offset(offs_sw_m, offs_sw_n, stride_0=num_pid_n)
    mask_sw = get_2d_mask(offs_sw_m, offs_sw_n, max_0=num_pid_m, max_1=num_pid_n)
    
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(z_ptr + offs_sw, x, mask=mask_sw)

@triton.jit
def swizzle_k_1d(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n 

    pid_m_, pid_n_ = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)  # Weirdness: tl.swizzle2d doesn't work when simulating on CPU
    
    offs_m = get_1d_offset(1, n_prev_chunks=pid_m)
    offs_n = get_1d_offset(1, n_prev_chunks=pid_n)
    
    offs = get_2d_offset(offs_m, offs_n, stride_0=num_pid_n)
    mask = get_2d_mask(offs_m, offs_n, max_0=num_pid_m, max_1=num_pid_n)

    offs_sw_m = get_1d_offset(1, n_prev_chunks=pid_m_)
    offs_sw_n = get_1d_offset(1, n_prev_chunks=pid_n_)
    
    offs_sw = get_2d_offset(offs_sw_m, offs_sw_n, stride_0=num_pid_n)
    mask_sw = get_2d_mask(offs_sw_m, offs_sw_n, max_0=num_pid_m, max_1=num_pid_n)
    
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(z_ptr + offs_sw, x, mask=mask_sw)

@triton.jit
def column_major(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n 

    pid_m_, pid_n_ = col_major(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    
    offs_m = get_1d_offset(1, n_prev_chunks=pid_m)
    offs_n = get_1d_offset(1, n_prev_chunks=pid_n)
    
    offs = get_2d_offset(offs_m, offs_n, stride_0=num_pid_n)
    mask = get_2d_mask(offs_m, offs_n, max_0=num_pid_m, max_1=num_pid_n)

    offs_sw_m = get_1d_offset(1, n_prev_chunks=pid_m_)
    offs_sw_n = get_1d_offset(1, n_prev_chunks=pid_n_)
    
    offs_sw = get_2d_offset(offs_sw_m, offs_sw_n, stride_0=num_pid_n)
    mask_sw = get_2d_mask(offs_sw_m, offs_sw_n, max_0=num_pid_m, max_1=num_pid_n)
    
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(z_ptr + offs_sw, x, mask=mask_sw)

print("Grouped scheduling")
out1 = z.clone()
grouped[(BLOCKS_M * BLOCKS_N,)](x, out1, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
print(out1.cpu().numpy())

print("Swizzle-2d (same as grouped) with 1D grid scheduling")
out1 = z.clone()
swizzle_k_1d[(BLOCKS_M * BLOCKS_N,)](x, out1, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
print(out1.cpu().numpy())

print("Swizzle-2d (same as grouped) with 2D grid scheduling")
out2 = z.clone()
swizzle_k_2d[(BLOCKS_M, BLOCKS_N)](x, out2, GROUP_SIZE_M)
print(out2.cpu().numpy())

print("Column Major scheduling")
out3 = z.clone()
column_major[(BLOCKS_M * BLOCKS_N,)](x, out3, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
print(out3.cpu().numpy())