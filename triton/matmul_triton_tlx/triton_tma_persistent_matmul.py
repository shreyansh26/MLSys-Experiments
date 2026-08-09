"""Persistent TMA matmul with grouped scheduling, epilogue subtiling, and WS."""

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
    c_block_n = block_n // 2 if args["EPILOGUE_SUBTILE"] else block_n
    args["c_desc"].block_shape = [block_m, c_block_n]


CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "GROUP_M": 8,
            "EPILOGUE_SUBTILE": subtile,
        },
        num_warps=warps,
        num_stages=stages,
        pre_hook=_descriptor_hook,
    )
    for bm, bn, bk, warps, stages, subtile in [
        (64, 128, 64, 4, 4, False),
        (128, 64, 64, 4, 4, False),
        (128, 128, 64, 8, 4, False),
        (128, 128, 64, 8, 5, True),
        (64, 256, 64, 8, 4, True),
        (128, 256, 64, 8, 4, True),
        (128, 128, 128, 8, 3, True),
        (256, 128, 64, 8, 3, True),
    ]
]


@triton.jit
def _grouped_tile(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    programs_per_group = GROUP_M * num_pid_n
    group_id = tile_id // programs_per_group
    first_pid_m = group_id * GROUP_M
    group_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % group_m)
    pid_n = (tile_id % programs_per_group) // group_m
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
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    # Duplicate the output counter to avoid a Blackwell prologue/epilogue
    # pipelining dependency, matching the official persistent tutorial.
    output_tile_id = start_pid - NUM_SMS
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=True,
        warp_specialize=WARP_SPECIALIZE,
    ):
        pid_m, pid_n = _grouped_tile(
            tile_id, num_pid_m, num_pid_n, GROUP_M
        )
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_k, offset_n])
            accumulator = tl.dot(a, b, accumulator)

        output_tile_id += NUM_SMS
        out_m, out_n = _grouped_tile(
            output_tile_id, num_pid_m, num_pid_n, GROUP_M
        )
        offset_cm = out_m * BLOCK_M
        offset_cn = out_n * BLOCK_N
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c_desc.store([offset_cm, offset_cn], acc0.to(c_desc.dtype))
            c_desc.store(
                [offset_cm, offset_cn + BLOCK_N // 2], acc1.to(c_desc.dtype)
            )
        else:
            c_desc.store(
                [offset_cm, offset_cn], accumulator.to(c_desc.dtype)
            )


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    warp_specialize: bool | None = None,
) -> torch.Tensor:
    if not supports_tma(a.device):
        raise RuntimeError("persistent TMA matmul requires Hopper or newer")
    warp_specialize = resolve_triton_warp_specialization(
        a.device, warp_specialize
    )
    a, b, m, n, k, original_n = aligned_inputs(a, b)
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    dummy = [1, 1]
    a_desc = TensorDescriptor.from_tensor(a, dummy)
    b_desc = TensorDescriptor.from_tensor(b, dummy)
    c_desc = TensorDescriptor.from_tensor(c, dummy)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    grid = lambda meta: (
        min(
            num_sms,
            triton.cdiv(m, meta["BLOCK_M"])
            * triton.cdiv(n, meta["BLOCK_N"]),
        ),
    )
    _kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M=m,
        N=n,
        K=k,
        NUM_SMS=num_sms,
        WARP_SPECIALIZE=warp_specialize,
    )
    return restore_output(c, original_n)


def warp_specialized_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return matmul(a, b, warp_specialize=True)
