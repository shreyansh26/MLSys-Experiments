"""Persistent Blackwell TLX GEMM using TMA, TMEM, and tcgen05 MMA."""

from __future__ import annotations

from tlx_plugin import load_tlx

tlx = load_tlx()

import torch
import triton
import triton.language as tl
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from reference import aligned_inputs, restore_output, supports_tma

BLOCK_M = tl.constexpr(128)
BLOCK_N = tl.constexpr(256)
BLOCK_K = tl.constexpr(64)
PIPELINE_STAGES = tl.constexpr(4)
ACCUMULATOR_STAGES = tl.constexpr(2)
EPILOGUE_SUBTILES = tl.constexpr(4)


def _shared_layout(shape: list[int], dtype: torch.dtype):
    gl_dtype = gl.float16 if dtype == torch.float16 else gl.bfloat16
    return gl.NVMMASharedLayout.get_default_for(shape, gl_dtype)


@triton.jit
def _buffer_state(iteration, num_buffers: tl.constexpr):
    return iteration % num_buffers, (iteration // num_buffers) & 1


@triton.jit
def _grouped_tile(
    tile_id,
    num_pid_m,
    num_pid_n,
    group_m: tl.constexpr,
):
    programs_per_group = group_m * num_pid_n
    group_id = tile_id // programs_per_group
    first_pid_m = group_id * group_m
    actual_group_m = min(num_pid_m - first_pid_m, group_m)
    tile_in_group = tile_id % programs_per_group
    return (
        first_pid_m + tile_in_group % actual_group_m,
        tile_in_group // actual_group_m,
    )


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
):
    input_dtype: tl.constexpr = a_ptr.dtype.element_ty
    a_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), input_dtype, PIPELINE_STAGES)
    b_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), input_dtype, PIPELINE_STAGES)
    accumulators = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        ACCUMULATOR_STAGES,
        storage=tlx.storage_kind.tmem,
    )

    load_empty = tlx.alloc_barriers(num_barriers=PIPELINE_STAGES, arrive_count=1)
    load_ready = tlx.alloc_barriers(num_barriers=PIPELINE_STAGES, arrive_count=1)
    acc_empty = tlx.alloc_barriers(num_barriers=ACCUMULATOR_STAGES, arrive_count=1)
    acc_ready = tlx.alloc_barriers(num_barriers=ACCUMULATOR_STAGES, arrive_count=1)

    # Three hardware-specialized partitions run concurrently:
    #   - one warp issues TMA loads,
    #   - one warp issues tcgen05 MMA into TMEM,
    #   - four default warps load TMEM subtiles and store the epilogue.
    with tlx.async_tasks():
        with tlx.async_task("default", registers=232):
            start_pid = tl.program_id(0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_tiles = num_pid_m * num_pid_n
            tile_id = start_pid
            tile_iteration = 0
            while tile_id < num_tiles:
                pid_m, pid_n = _grouped_tile(tile_id, num_pid_m, num_pid_n, 8)
                acc_slot, acc_phase = _buffer_state(tile_iteration, ACCUMULATOR_STAGES)
                tlx.barrier_wait(tlx.local_view(acc_ready, acc_slot), acc_phase)
                accumulator = tlx.local_view(accumulators, acc_slot)
                subtile_n: tl.constexpr = BLOCK_N // EPILOGUE_SUBTILES
                rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
                for subtile in tl.static_range(EPILOGUE_SUBTILES):
                    accumulator_subtile = tlx.local_slice(
                        accumulator,
                        [0, subtile * subtile_n],
                        [BLOCK_M, subtile_n],
                    )
                    output = tlx.local_load(accumulator_subtile)
                    cols = (
                        pid_n * BLOCK_N
                        + subtile * subtile_n
                        + tl.arange(0, subtile_n)[None, :]
                    )
                    mask = (rows < M) & (cols < N)
                    tl.store(
                        c_ptr + rows * N + cols,
                        output.to(input_dtype),
                        mask=mask,
                    )
                tlx.barrier_arrive(tlx.local_view(acc_empty, acc_slot))
                tile_iteration += 1
                tile_id += NUM_SMS

        with tlx.async_task(num_warps=1, registers=24):
            start_pid = tl.program_id(0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_tiles = num_pid_m * num_pid_n
            tile_id = start_pid
            load_iteration = 0
            while tile_id < num_tiles:
                pid_m, pid_n = _grouped_tile(tile_id, num_pid_m, num_pid_n, 8)
                offset_m = pid_m * BLOCK_M
                offset_n = pid_n * BLOCK_N
                for k_tile in range(tl.cdiv(K, BLOCK_K)):
                    slot, phase = _buffer_state(load_iteration, PIPELINE_STAGES)
                    empty = tlx.local_view(load_empty, slot)
                    ready = tlx.local_view(load_ready, slot)
                    tlx.barrier_wait(empty, phase ^ 1)
                    tlx.barrier_expect_bytes(
                        ready,
                        (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N)
                        * tlx.size_of(input_dtype),
                    )
                    offset_k = k_tile * BLOCK_K
                    tlx.async_descriptor_load(
                        a_desc,
                        tlx.local_view(a_buffers, slot),
                        [offset_m, offset_k],
                        ready,
                    )
                    tlx.async_descriptor_load(
                        b_desc,
                        tlx.local_view(b_buffers, slot),
                        [offset_k, offset_n],
                        ready,
                    )
                    load_iteration += 1
                tile_id += NUM_SMS

        with tlx.async_task(num_warps=1, registers=24):
            start_pid = tl.program_id(0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_tiles = num_pid_m * num_pid_n
            tile_id = start_pid
            load_iteration = 0
            tile_iteration = 0
            while tile_id < num_tiles:
                acc_slot, acc_phase = _buffer_state(tile_iteration, ACCUMULATOR_STAGES)
                tlx.barrier_wait(tlx.local_view(acc_empty, acc_slot), acc_phase ^ 1)
                accumulator = tlx.local_view(accumulators, acc_slot)
                for k_tile in range(tl.cdiv(K, BLOCK_K)):
                    slot, phase = _buffer_state(load_iteration, PIPELINE_STAGES)
                    tlx.barrier_wait(tlx.local_view(load_ready, slot), phase)
                    tlx.async_dot(
                        tlx.local_view(a_buffers, slot),
                        tlx.local_view(b_buffers, slot),
                        accumulator,
                        use_acc=k_tile != 0,
                    )
                    tlx.tcgen05_commit(tlx.local_view(load_empty, slot))
                    load_iteration += 1
                tlx.tcgen05_commit(tlx.local_view(acc_ready, acc_slot))
                tile_iteration += 1
                tile_id += NUM_SMS


def _allocator(size: int, alignment: int, stream: int | None):
    del alignment, stream
    return torch.empty(size, dtype=torch.int8, device="cuda")


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not supports_tma(a.device) or torch.cuda.get_device_capability(a.device)[0] < 10:
        raise RuntimeError("this TLX TCGen5 kernel requires Blackwell (SM100+)")
    a, b, m, n, k, original_n = aligned_inputs(a, b)
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    triton.set_allocator(_allocator)

    block_m, block_n, block_k = 128, 256, 64
    a_shape = [block_m, block_k]
    b_shape = [block_k, block_n]
    a_desc = TensorDescriptor.from_tensor(a, a_shape, _shared_layout(a_shape, a.dtype))
    b_desc = TensorDescriptor.from_tensor(b, b_shape, _shared_layout(b_shape, b.dtype))
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (
        min(
            num_sms,
            triton.cdiv(m, block_m) * triton.cdiv(n, block_n),
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
        num_warps=4,
    )
    return restore_output(c, original_n)
