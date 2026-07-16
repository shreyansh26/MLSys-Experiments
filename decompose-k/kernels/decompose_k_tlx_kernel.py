"""Single-launch TLX Decompose-K kernel for Hopper GPUs.

Each split computes with staged asynchronous loads and WGMMA, atomically
reduces its fp32 tile into a shared workspace, and lets the last arriving split
write the output and reset the workspace for the next launch.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tlx_plugin import load_tlx

tlx = load_tlx()

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


K_SPLITS = (2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class TLXConfig:
    split_k: int
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int


@triton.jit
def _decompose_k_tlx_atomic(
    a,
    b,
    accumulator_desc,
    counters,
    c,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    FUSE_RELU: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_per_split: tl.constexpr = K // SPLIT_K
    split_start = split_id * k_per_split

    a_ptrs = (
        a
        + offs_m[:, None] * stride_am
        + (split_start + offs_k[None, :]) * stride_ak
    )
    b_ptrs = (
        b
        + (split_start + offs_k[:, None]) * stride_bk
        + offs_n[None, :] * stride_bn
    )

    buffers_a = tlx.local_alloc(
        (BLOCK_M, BLOCK_K), tlx.dtype_of(a), NUM_STAGES
    )
    buffers_b = tlx.local_alloc(
        (BLOCK_K, BLOCK_N), tlx.dtype_of(b), NUM_STAGES
    )

    for stage in tl.range(
        0,
        NUM_STAGES - 1,
        loop_unroll_factor=NUM_STAGES - 1,
    ):
        a_stage = tlx.local_view(buffers_a, stage)
        b_stage = tlx.local_view(buffers_b, stage)
        k_mask = offs_k < k_per_split - stage * BLOCK_K
        token_a = tlx.async_load(
            a_ptrs,
            a_stage,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
        )
        token_b = tlx.async_load(
            b_ptrs,
            b_stage,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
        )
        tlx.async_load_commit_group([token_a, token_b])
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_iters: tl.constexpr = tl.cdiv(k_per_split, BLOCK_K)
    if num_k_iters == 1:
        a_stage = tlx.local_view(buffers_a, 0)
        b_stage = tlx.local_view(buffers_b, 0)
        tlx.async_load_wait_group(0)
        acc = tlx.async_dot(a_stage, b_stage, acc)
    else:
        for k_iter in tl.range(0, num_k_iters, num_stages=0):
            buffer_index = k_iter % NUM_STAGES
            a_stage = tlx.local_view(buffers_a, buffer_index)
            b_stage = tlx.local_view(buffers_b, buffer_index)
            tlx.async_load_wait_group(NUM_STAGES - 2)
            acc = tlx.async_dot(a_stage, b_stage, acc)

            next_iter = k_iter + NUM_STAGES - 1
            next_index = next_iter % NUM_STAGES
            a_next = tlx.local_view(buffers_a, next_index)
            b_next = tlx.local_view(buffers_b, next_index)
            acc = tlx.async_dot_wait(1, acc)
            k_mask = offs_k < k_per_split - next_iter * BLOCK_K
            token_a = tlx.async_load(
                a_ptrs,
                a_next,
                mask=(offs_m[:, None] < M) & k_mask[None, :],
            )
            token_b = tlx.async_load(
                b_ptrs,
                b_next,
                mask=k_mask[:, None] & (offs_n[None, :] < N),
            )
            tlx.async_load_commit_group([token_a, token_b])
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    acc = tlx.async_dot_wait(0, acc)
    accumulator_desc.atomic_add(
        [pid_m * BLOCK_M, pid_n * BLOCK_N],
        acc,
    )

    previous = tl.atomic_add(
        counters + pid,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    if previous == SPLIT_K - 1:
        result = accumulator_desc.load(
            [pid_m * BLOCK_M, pid_n * BLOCK_N],
        )
        if FUSE_RELU:
            result = tl.maximum(result, 0.0)
        c_ptrs = c + offs_m[:, None] * N + offs_n[None, :]
        tl.store(
            c_ptrs,
            result,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )
        accumulator_desc.store(
            [pid_m * BLOCK_M, pid_n * BLOCK_N],
            tl.zeros((BLOCK_M, BLOCK_N), tl.float32),
        )
        tl.atomic_xchg(
            counters + pid,
            0,
            sem="release",
            scope="gpu",
        )


def inductor_like_splits(m: int, n: int, k: int, limit: int) -> list[int]:
    max_split = min(k // m, k // n)
    power_splits = [
        split
        for split in K_SPLITS
        if split <= max_split and k % split == 0 and k // split >= 16
    ]
    pow2_k_parts = []
    multiple_32_k_parts = []
    rest = []
    for split in range(2, max_split + 1):
        if k % split != 0 or split in power_splits:
            continue
        k_part = k // split
        if k_part < 128:
            continue
        if k_part & (k_part - 1) == 0:
            pow2_k_parts.append(split)
        elif k_part % 32 == 0:
            multiple_32_k_parts.append(split)
        else:
            rest.append(split)
    extra_splits = (pow2_k_parts + multiple_32_k_parts + rest)[:limit]
    return power_splits + extra_splits


def candidate_configs(split_values: list[int]) -> list[TLXConfig]:
    tile_configs = [
        (64, 32, 128, 4, 2),
        (64, 32, 256, 4, 2),
        (64, 32, 512, 4, 2),
    ]
    return [
        TLXConfig(
            split,
            block_m,
            block_n,
            block_k,
            8,
            num_warps,
            num_stages,
        )
        for split in split_values
        for block_m, block_n, block_k, num_warps, num_stages in tile_configs
    ]


def workspace_elements(m: int, n: int, config: TLXConfig) -> int:
    num_tiles = triton.cdiv(m, config.block_m) * triton.cdiv(
        n, config.block_n
    )
    return m * n + num_tiles


def _initialize_workspace_once(workspace: torch.Tensor) -> None:
    if getattr(workspace, "_tlx_atomic_ready", False):
        return
    workspace.zero_()
    workspace._tlx_atomic_ready = True


def _run(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    workspace: torch.Tensor,
    config: TLXConfig,
    *,
    fuse_relu: bool,
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("the TLX kernel currently supports fp16 and bf16")
    if a.dtype != b.dtype or c.dtype != a.dtype:
        raise ValueError("a, b, and c must use the same dtype")
    if (
        a.device != b.device
        or a.device != c.device
        or a.device != workspace.device
    ):
        raise ValueError("a, b, c, and workspace must be on the same device")
    if b.shape[0] != k or c.shape != (m, n):
        raise ValueError("incompatible matmul or output shapes")
    if k % config.split_k != 0:
        raise ValueError(f"K={k} must be divisible by split_k={config.split_k}")
    if workspace.dtype != torch.float32:
        raise ValueError("workspace must use fp32 storage")
    if not workspace.is_contiguous() or not c.is_contiguous():
        raise ValueError("the TLX kernel requires contiguous outputs")
    if workspace.numel() < workspace_elements(m, n, config):
        raise ValueError("workspace does not have enough storage")
    _initialize_workspace_once(workspace)

    num_tiles = triton.cdiv(m, config.block_m) * triton.cdiv(
        n, config.block_n
    )
    accumulator = workspace.flatten()[: m * n].view(m, n)
    counters = workspace.view(torch.int32).flatten()[m * n : m * n + num_tiles]
    accumulator_desc = TensorDescriptor.from_tensor(
        accumulator,
        [config.block_m, config.block_n],
    )
    grid = (num_tiles, config.split_k)
    _decompose_k_tlx_atomic[grid](
        a,
        b,
        accumulator_desc,
        counters,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        M=m,
        N=n,
        K=k,
        SPLIT_K=config.split_k,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        GROUP_M=config.group_m,
        NUM_STAGES=config.num_stages,
        FUSE_RELU=fuse_relu,
        num_warps=config.num_warps,
        num_stages=1,
    )
    return c


def decompose_k_relu_out(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    workspace: torch.Tensor,
    config: TLXConfig,
    *,
    fuse_relu: bool,
) -> torch.Tensor:
    return _run(a, b, c, workspace, config, fuse_relu=fuse_relu)


def decompose_k_matmul_out(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    workspace: torch.Tensor,
    config: TLXConfig,
) -> torch.Tensor:
    return _run(a, b, c, workspace, config, fuse_relu=False)
