"""Autotuned one-program-per-output-tile TMA matmul with a Blackwell WS path."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from reference import (
    aligned_inputs,
    resolve_triton_warp_specialization,
    restore_output,
    supports_tma,
)


def _descriptor_hook(args) -> None:
    block_m = args["BLOCK_M"]
    block_n = args["BLOCK_N"]
    block_k = args["BLOCK_K"]
    args["a_desc"].block_shape = [block_m, block_k]
    args["b_desc"].block_shape = [block_k, block_n]
    args["c_desc"].block_shape = [block_m, block_n]


CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
        num_warps=warps,
        num_stages=stages,
        pre_hook=_descriptor_hook,
    )
    for bm, bn, bk, warps, stages in [
        (64, 64, 64, 4, 3),
        (64, 128, 64, 4, 4),
        (128, 64, 64, 4, 4),
        (128, 128, 64, 8, 4),
        (64, 256, 64, 8, 3),
        (128, 256, 64, 8, 3),
        (128, 128, 128, 8, 3),
        (256, 128, 64, 8, 3),
    ]
]


@triton.jit
def _grouped_tile(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    programs_per_group = GROUP_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_M
    group_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_m)
    pid_n = (pid % programs_per_group) // group_m
    return pid_m, pid_n


@triton.autotune(configs=CONFIGS, key=["M", "N", "K", "WARP_SPECIALIZE"])
@triton.jit
def _kernel(
    a_desc,
    b_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    pid_m, pid_n = _grouped_tile(
        tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M
    )
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # On Blackwell this forms dedicated producer/consumer warp partitions.
    # On Hopper WARP_SPECIALIZE must be False and num_stages pipelines the loop.
    for k_tile in tl.range(
        0, tl.cdiv(K, BLOCK_K), warp_specialize=WARP_SPECIALIZE
    ):
        offset_k = k_tile * BLOCK_K
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_k, offset_n])
        accumulator = tl.dot(a, b, accumulator)

    c_desc.store([offset_m, offset_n], accumulator.to(c_desc.dtype))


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    warp_specialize: bool | None = None,
) -> torch.Tensor:
    if not supports_tma(a.device):
        raise RuntimeError("TMA matmul requires an NVIDIA Hopper GPU or newer")
    warp_specialize = resolve_triton_warp_specialization(
        a.device, warp_specialize
    )
    a, b, m, n, k, original_n = aligned_inputs(a, b)
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    dummy = [1, 1]
    a_desc = TensorDescriptor.from_tensor(a, dummy)
    b_desc = TensorDescriptor.from_tensor(b, dummy)
    c_desc = TensorDescriptor.from_tensor(c, dummy)

    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M=m,
        N=n,
        K=k,
        WARP_SPECIALIZE=warp_specialize,
    )
    return restore_output(c, original_n)


def warp_specialized_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return matmul(a, b, warp_specialize=True)
