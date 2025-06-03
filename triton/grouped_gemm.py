import torch

import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148

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

                # load tile
                a = tl.load(a_ptrs, mask=(offset_am[:, None] < gm) & (offset_k[None, :] < gk), other=0.0)
                b = tl.load(b_ptrs, mask=(offset_k[:, None] < gk) & (offset_bn[None, :] < gn), other=0.0)

                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * stride_b

            c = accumulator.to(tl.float16)

            offset_cm = m_tile_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offset_cn = n_tile_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            c_ptrs = c_ptr + offset_cm[:, None] * stride_c + offset_cn[None, :]

            tl.store(c_ptrs, c)

            # go to the next tile - somewhat like grid-stride loop
            tile_idx += NUM_SM

        # go to the next GEMM
        last_problem_end += num_tiles       