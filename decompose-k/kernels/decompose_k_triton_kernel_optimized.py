"""Optimized Triton Decompose-K matmul with vectorized split reduction.

This keeps the baseline ``kernels.decompose_k_triton_kernel`` implementation
intact and adds a separate candidate optimized around small-M/N, large-K
matmuls with explicit Decompose-K partials.
"""

import torch
import triton
import triton.language as tl

from kernels.decompose_k_triton_kernel import (
    KernelConfig,
    _check_inputs,
    _partial_mm,
    inductor_like_splits as _baseline_splits,
)


K_SPLITS = (2, 4, 8, 16, 32, 64, 128, 256)


@triton.jit
def _reduce_epilogue_vector(
    partials,
    c,
    stride_ps: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    FUSE_RELU: tl.constexpr,
):
    x_base = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    x = x_base[:, None]
    r = tl.arange(0, RBLOCK)[None, :]
    xmask = x < (M * N)
    rmask = r < SPLIT_K
    offs_m = x // N
    offs_n = x - offs_m * N
    partial_offsets = offs_m * stride_pm + offs_n * stride_pn
    vals = tl.load(
        partials + r * stride_ps + partial_offsets,
        mask=xmask & rmask,
        other=0.0,
    )
    acc = tl.sum(vals, 1)

    if FUSE_RELU:
        acc = tl.maximum(acc, 0.0)

    store_m = x_base // N
    store_n = x_base - store_m * N
    c_offsets = store_m * stride_cm + store_n * stride_cn
    tl.store(c + c_offsets, acc, mask=x_base < (M * N))


@triton.jit
def _reduce_epilogue_vector_flat(
    partials,
    c,
    stride_ps: tl.constexpr,
    XNUMEL: tl.constexpr,
    SPLIT_K: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    FUSE_RELU: tl.constexpr,
):
    x_base = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    x = x_base[:, None]
    r = tl.arange(0, RBLOCK)[None, :]
    vals = tl.load(
        partials + r * stride_ps + x,
        mask=(x < XNUMEL) & (r < SPLIT_K),
        other=0.0,
    )
    acc = tl.sum(vals, 1)

    if FUSE_RELU:
        acc = tl.maximum(acc, 0.0)

    tl.store(c + x_base, acc, mask=x_base < XNUMEL)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _xblock(split_k: int) -> int:
    if split_k <= 64:
        return 32
    if split_k <= 128:
        return 16
    return 8


def inductor_like_splits(m: int, n: int, k: int, limit: int) -> list[int]:
    """Include custom-op-style power splits plus the original heuristic splits."""
    del limit
    max_split = min(k // m, k // n)
    splits = []
    for split in K_SPLITS:
        if split <= max_split and k % split == 0 and k // split >= 16:
            splits.append(split)
    for split in _baseline_splits(m, n, k, 16):
        if split not in splits:
            splits.append(split)
    return splits


def candidate_configs(split_values: list[int]) -> list[KernelConfig]:
    """Search BMM-like partial tiles plus the vectorized split reducer."""
    small_tiles = [
        (16, 16, 32, 1, 4),
        (16, 16, 64, 2, 4),
        (16, 16, 64, 4, 4),
        (16, 16, 64, 1, 5),
        (16, 16, 128, 2, 4),
        (16, 16, 128, 4, 4),
        (16, 16, 128, 1, 4),
        (16, 16, 128, 1, 5),
        (16, 32, 64, 1, 5),
        (16, 32, 128, 1, 4),
        (32, 16, 64, 1, 5),
        (32, 16, 128, 1, 4),
        (32, 32, 64, 2, 4),
        (32, 32, 128, 2, 4),
    ]
    large_tiles = [
        (64, 32, 32, 4, 5),
        (64, 32, 64, 4, 5),
        (64, 32, 128, 4, 5),
        (32, 64, 32, 4, 5),
        (32, 64, 64, 4, 5),
        (32, 64, 128, 4, 5),
        (64, 64, 32, 4, 3),
        (64, 64, 32, 4, 4),
        (64, 64, 64, 4, 3),
        (64, 64, 128, 4, 4),
        (64, 64, 128, 4, 5),
        (64, 64, 64, 8, 5),
        (64, 64, 128, 8, 4),
    ]

    configs = []
    seen = set()
    for split_k in split_values:
        for block_m, block_n, block_k, num_warps, num_stages in [*small_tiles, *large_tiles]:
            group_m = 8
            key = (split_k, block_m, block_n, block_k, num_warps, num_stages)
            if key in seen:
                continue
            seen.add(key)
            configs.append(
                KernelConfig(
                    split_k,
                    block_m,
                    block_n,
                    block_k,
                    group_m,
                    num_warps,
                    num_stages,
                )
            )
    return configs


def decompose_k_relu_out(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    partials: torch.Tensor,
    config: KernelConfig,
    *,
    fuse_relu: bool,
) -> torch.Tensor:
    _check_inputs(a, b, config.split_k)
    m, k = a.shape
    n = b.shape[1]

    partial_grid = (
        triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),
        config.split_k,
    )
    _partial_mm[partial_grid](
        a,
        b,
        partials,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        M=m,
        N=n,
        K=k,
        SPLIT_K=config.split_k,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        GROUP_M=config.group_m,
        INPUT_PRECISION="ieee" if a.dtype is torch.float32 else "tf32",
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    xblock = _xblock(config.split_k)
    rblock = _next_power_of_2(config.split_k)
    reduce_grid = (triton.cdiv(m * n, xblock),)
    if partials.is_contiguous() and c.is_contiguous():
        _reduce_epilogue_vector_flat[reduce_grid](
            partials,
            c,
            partials.stride(0),
            XNUMEL=m * n,
            SPLIT_K=config.split_k,
            XBLOCK=xblock,
            RBLOCK=rblock,
            FUSE_RELU=fuse_relu,
            num_warps=1,
            num_stages=3,
        )
    else:
        _reduce_epilogue_vector[reduce_grid](
            partials,
            c,
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            c.stride(0),
            c.stride(1),
            M=m,
            N=n,
            SPLIT_K=config.split_k,
            XBLOCK=xblock,
            RBLOCK=rblock,
            FUSE_RELU=fuse_relu,
            num_warps=1,
            num_stages=3,
        )
    return c


def decompose_k_matmul_out(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    partials: torch.Tensor,
    config: KernelConfig,
) -> torch.Tensor:
    return decompose_k_relu_out(a, b, c, partials, config, fuse_relu=False)
