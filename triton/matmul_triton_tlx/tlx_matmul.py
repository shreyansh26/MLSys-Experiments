"""Persistent Hopper TLX GEMM with a TMA producer and two WGMMA consumers."""

from __future__ import annotations

from tlx_plugin import load_tlx

tlx = load_tlx()

import torch
import triton
import triton.language as tl
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from reference import aligned_inputs, restore_output, supports_tma


def _descriptor_hook(args) -> None:
    block_m_split = args["BLOCK_M"] // args["NUM_MMA_GROUPS"]
    block_n = args["BLOCK_N"]
    block_k = args["BLOCK_K"]
    args["a_desc"].block_shape = [block_m_split, block_k]
    args["b_desc"].block_shape = [block_k, block_n]


def _shared_layout(shape: list[int], dtype: torch.dtype):
    gl_dtype = gl.float16 if dtype == torch.float16 else gl.bfloat16
    return gl.NVMMASharedLayout.get_default_for(shape, gl_dtype)


CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "PIPELINE_STAGES": stages,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": subtile,
        },
        num_warps=4,
        num_stages=1,
        pre_hook=_descriptor_hook,
    )
    for bm, bn, stages, subtile in [
        (128, 128, 3, False),
        (128, 128, 4, True),
        (128, 256, 3, True),
        (128, 256, 4, True),
        (256, 128, 3, True),
        (256, 128, 4, True),
    ]
]


@triton.jit
def _buffer_state(iteration, NUM_BUFFERS: tl.constexpr):
    index = iteration % NUM_BUFFERS
    phase = (iteration // NUM_BUFFERS) & 1
    return index, phase


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


@triton.autotune(configs=CONFIGS, key=["M", "N", "K"])
@triton.jit
def _kernel(
    a_ptr,
    a_desc,
    b_desc,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):
    # A raw tensor argument keeps stock Triton's device/plugin dispatch on the
    # TLX compilation path; all actual movement still uses TMA descriptors.
    input_dtype: tl.constexpr = a_ptr.dtype.element_ty
    block_m_split: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    a_buffers = tlx.local_alloc(
        (block_m_split, BLOCK_K),
        input_dtype,
        PIPELINE_STAGES * NUM_MMA_GROUPS,
    )
    b_buffers = tlx.local_alloc(
        (BLOCK_K, BLOCK_N), input_dtype, PIPELINE_STAGES
    )

    # Each A buffer has one consumer. Both consumer warpgroups share B.
    empty_a = tlx.alloc_barriers(
        num_barriers=PIPELINE_STAGES * NUM_MMA_GROUPS, arrive_count=1
    )
    empty_b = tlx.alloc_barriers(
        num_barriers=PIPELINE_STAGES, arrive_count=NUM_MMA_GROUPS
    )
    full_a = tlx.alloc_barriers(
        num_barriers=PIPELINE_STAGES * NUM_MMA_GROUPS, arrive_count=1
    )
    full_b = tlx.alloc_barriers(
        num_barriers=PIPELINE_STAGES, arrive_count=1
    )

    # This is explicit warp specialization: the default producer only issues
    # TMA, while two replicated four-warp consumers only issue WGMMA/stores.
    with tlx.async_tasks():
        with tlx.async_task("default", registers=24):
            start_pid = tl.program_id(0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_tiles = num_pid_m * num_pid_n
            tile_id = start_pid
            iteration = 0
            while tile_id < num_tiles:
                pid_m, pid_n = _grouped_tile(
                    tile_id, num_pid_m, num_pid_n, GROUP_M
                )
                offset_m = pid_m * BLOCK_M
                offset_n = pid_n * BLOCK_N
                for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
                    slot, phase = _buffer_state(iteration, PIPELINE_STAGES)
                    offset_k = k_tile * BLOCK_K

                    b_empty = tlx.local_view(empty_b, slot)
                    b_full = tlx.local_view(full_b, slot)
                    tlx.barrier_wait(b_empty, phase ^ 1)
                    tlx.barrier_expect_bytes(
                        b_full,
                        BLOCK_K
                        * BLOCK_N
                        * tlx.size_of(input_dtype),
                    )
                    tlx.async_descriptor_load(
                        b_desc,
                        tlx.local_view(b_buffers, slot),
                        [offset_k, offset_n],
                        b_full,
                    )

                    for group in tl.static_range(NUM_MMA_GROUPS):
                        a_slot = slot + group * PIPELINE_STAGES
                        a_empty = tlx.local_view(empty_a, a_slot)
                        a_full = tlx.local_view(full_a, a_slot)
                        tlx.barrier_wait(a_empty, phase ^ 1)
                        tlx.barrier_expect_bytes(
                            a_full,
                            block_m_split
                            * BLOCK_K
                            * tlx.size_of(input_dtype),
                        )
                        tlx.async_descriptor_load(
                            a_desc,
                            tlx.local_view(a_buffers, a_slot),
                            [offset_m + group * block_m_split, offset_k],
                            a_full,
                        )
                    iteration += 1
                tile_id += NUM_SMS

        with tlx.async_task(num_warps=4, replicate=2, registers=232):
            consumer: tl.constexpr = tlx.async_task_replica_id()
            start_pid = tl.program_id(0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_tiles = num_pid_m * num_pid_n
            tile_id = start_pid
            iteration = 0
            while tile_id < num_tiles:
                pid_m, pid_n = _grouped_tile(
                    tile_id, num_pid_m, num_pid_n, GROUP_M
                )
                accumulator = tl.zeros(
                    (block_m_split, BLOCK_N), dtype=tl.float32
                )
                for _ in range(0, tl.cdiv(K, BLOCK_K)):
                    slot, phase = _buffer_state(iteration, PIPELINE_STAGES)
                    a_slot = slot + consumer * PIPELINE_STAGES
                    a_full = tlx.local_view(full_a, a_slot)
                    b_full = tlx.local_view(full_b, slot)
                    tlx.barrier_wait(a_full, phase)
                    tlx.barrier_wait(b_full, phase)
                    accumulator = tlx.async_dot(
                        tlx.local_view(a_buffers, a_slot),
                        tlx.local_view(b_buffers, slot),
                        accumulator,
                    )
                    accumulator = tlx.async_dot_wait(0, accumulator)
                    tlx.barrier_arrive(tlx.local_view(empty_a, a_slot))
                    tlx.barrier_arrive(tlx.local_view(empty_b, slot))
                    iteration += 1

                offset_m = (
                    pid_m * BLOCK_M + consumer * block_m_split
                )
                offset_n = pid_n * BLOCK_N
                if EPILOGUE_SUBTILE:
                    acc = tl.reshape(
                        accumulator,
                        (block_m_split, 2, BLOCK_N // 2),
                    )
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    rows = offset_m + tl.arange(0, block_m_split)[:, None]
                    cols = offset_n + tl.arange(0, BLOCK_N // 2)[None, :]
                    mask = (rows < M) & (cols < N)
                    tl.store(c_ptr + rows * N + cols, acc0.to(input_dtype), mask)
                    cols += BLOCK_N // 2
                    mask = (rows < M) & (cols < N)
                    tl.store(c_ptr + rows * N + cols, acc1.to(input_dtype), mask)
                else:
                    rows = offset_m + tl.arange(0, block_m_split)[:, None]
                    cols = offset_n + tl.arange(0, BLOCK_N)[None, :]
                    mask = (rows < M) & (cols < N)
                    tl.store(
                        c_ptr + rows * N + cols,
                        accumulator.to(input_dtype),
                        mask,
                    )
                tile_id += NUM_SMS


def _allocator(size: int, alignment: int, stream: int | None):
    del alignment, stream
    return torch.empty(size, dtype=torch.int8, device="cuda")


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not supports_tma(a.device) or torch.cuda.get_device_capability(a.device)[0] != 9:
        raise RuntimeError("this TLX WGMMA kernel targets Hopper (SM90)")
    a, b, m, n, k, original_n = aligned_inputs(a, b)
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    triton.set_allocator(_allocator)
    # The Gluon host descriptor carries the NVMMASharedLayout in its mangled
    # type. TLX's native async TMA verifier requires this layout to exactly
    # match the shared-memory destination selected by local_alloc.
    a_block = [64, 64]
    b_block = [64, 128]
    a_desc = TensorDescriptor.from_tensor(
        a, a_block, _shared_layout(a_block, a.dtype)
    )
    b_desc = TensorDescriptor.from_tensor(
        b, b_block, _shared_layout(b_block, b.dtype)
    )
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = lambda meta: (
        min(
            num_sms,
            triton.cdiv(m, meta["BLOCK_M"])
            * triton.cdiv(n, meta["BLOCK_N"]),
        ),
    )
    _kernel[grid](
        a,
        a_desc,
        b_desc,
        c,
        M=m,
        N=n,
        K=k,
        NUM_SMS=num_sms,
    )
    return restore_output(c, original_n)
