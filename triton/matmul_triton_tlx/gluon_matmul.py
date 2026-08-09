"""Persistent Hopper Gluon GEMM with TMA/WGMMA warp specialization."""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.language.core import _aggregate as aggregate

from reference import aligned_inputs, restore_output, supports_tma


@gluon.constexpr_function
def _warps_per_cta(block_m, block_n, num_warps):
    warps = [4, 1]
    while warps[0] * warps[1] != num_warps:
        if block_m > 16 * warps[0]:
            warps[0] *= 2
        else:
            warps[1] *= 2
    return warps


@gluon.constexpr_function
def _instruction_n(block_m, block_n, num_warps):
    m_repetitions = triton.cdiv(block_m, 16)
    n_repetitions = triton.cdiv(num_warps, m_repetitions)
    maximum_n = max(block_n // n_repetitions, 8)
    instruction_n = 256
    while instruction_n > maximum_n or block_n % instruction_n != 0:
        instruction_n -= 8
    return instruction_n


@gluon.constexpr_function
def _wgmma_layout(dtype, block_m, block_n, num_warps):
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=_warps_per_cta(block_m, block_n, num_warps),
        instr_shape=[
            16,
            _instruction_n(block_m, block_n, num_warps),
            256 // dtype.primitive_bitwidth,
        ],
    )


@aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    size: gl.constexpr

    @gluon.jit
    def create(phase, size: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), size)

    @gluon.must_use_result
    @gluon.jit
    def next(self):
        next_index = self.index + 1
        rollover = next_index == self.size
        return Counter(
            gl.where(rollover, 0, next_index),
            gl.where(rollover, self.phase ^ 1, self.phase),
            self.size,
        )


@aggregate
class GroupedPersistentScheduler:
    start_tile: gl.tensor
    tiles_m: gl.tensor
    tiles_per_group: gl.tensor
    tile_count: gl.tensor

    @gluon.jit
    def initialize(m, n, block_m: gl.constexpr, block_n: gl.constexpr):
        tiles_m = gl.cdiv(m, block_m)
        tiles_n = gl.cdiv(n, block_n)
        group_m: gl.constexpr = 8
        return GroupedPersistentScheduler(
            gl.program_id(0),
            tiles_m,
            group_m * tiles_n,
            tiles_m * tiles_n,
        )

    @gluon.jit
    def get_num_tiles(self):
        remaining = self.tile_count - self.start_tile
        return gl.cdiv(remaining, gl.num_programs(0))

    @gluon.jit
    def get_tile(self, iteration):
        group_m: gl.constexpr = 8
        tile = self.start_tile + iteration * gl.num_programs(0)
        group = tile // self.tiles_per_group
        first_m = group * group_m
        actual_group_m = min(self.tiles_m - first_m, group_m)
        tile_in_group = tile % self.tiles_per_group
        pid_m = first_m + tile_in_group % actual_group_m
        pid_n = tile_in_group // actual_group_m
        return pid_m, pid_n


@aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_buffers: gl.shared_memory_descriptor
    b_buffers: gl.shared_memory_descriptor
    empty_barriers: gl.shared_memory_descriptor
    ready_barriers: gl.shared_memory_descriptor
    c_buffer: gl.shared_memory_descriptor
    num_warps: gl.constexpr


@gluon.jit
def _load_partition(p):
    block_m: gl.constexpr = p.a_desc.block_type.shape[0]
    block_n: gl.constexpr = p.b_desc.block_type.shape[1]
    block_k: gl.constexpr = p.a_desc.block_type.shape[1]
    num_buffers: gl.constexpr = p.a_buffers.shape[0]
    state = Counter.create(1, num_buffers)
    scheduler = GroupedPersistentScheduler.initialize(
        p.c_desc.shape[0], p.c_desc.shape[1], block_m, block_n
    )

    for tile_index in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(tile_index)
        offset_m = pid_m * block_m
        offset_n = pid_n * block_n
        for offset_k in range(0, p.a_desc.shape[1], block_k):
            empty = p.empty_barriers.index(state.index)
            ready = p.ready_barriers.index(state.index)
            mbarrier.wait(empty, state.phase)
            mbarrier.expect(
                ready,
                p.a_desc.block_type.nbytes + p.b_desc.block_type.nbytes,
            )
            tma.async_copy_global_to_shared(
                p.a_desc,
                [offset_m, offset_k],
                ready,
                p.a_buffers.index(state.index),
            )
            tma.async_copy_global_to_shared(
                p.b_desc,
                [offset_k, offset_n],
                ready,
                p.b_buffers.index(state.index),
            )
            state = state.next()


@gluon.jit
def _compute_partition(p):
    block_m: gl.constexpr = p.a_desc.block_type.shape[0]
    block_n: gl.constexpr = p.b_desc.block_type.shape[1]
    block_k: gl.constexpr = p.a_desc.block_type.shape[1]
    num_buffers: gl.constexpr = p.a_buffers.shape[0]
    layout: gl.constexpr = _wgmma_layout(
        p.a_desc.dtype, block_m, block_n, p.num_warps
    )
    state = Counter.create(0, num_buffers)
    scheduler = GroupedPersistentScheduler.initialize(
        p.c_desc.shape[0], p.c_desc.shape[1], block_m, block_n
    )

    for tile_index in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(tile_index)
        accumulator = gl.zeros(
            (block_m, block_n), dtype=gl.float32, layout=layout
        )
        use_accumulator = gl.to_tensor(False)
        for _ in range(0, p.a_desc.shape[1], block_k):
            ready = p.ready_barriers.index(state.index)
            mbarrier.wait(ready, state.phase)
            accumulator = warpgroup_mma(
                p.a_buffers.index(state.index),
                p.b_buffers.index(state.index),
                accumulator,
                is_async=True,
                use_acc=use_accumulator,
            )
            accumulator = warpgroup_mma_wait(0, (accumulator,))
            mbarrier.arrive(p.empty_barriers.index(state.index), count=1)
            state = state.next()
            use_accumulator = gl.to_tensor(True)

        # Waiting here lets the previous tile's TMA store overlap this tile's
        # K loop without allowing its shared-memory source to be overwritten.
        tma.store_wait(pendings=0)
        p.c_buffer.store(accumulator.to(p.c_desc.dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(
            p.c_desc,
            [pid_m * block_m, pid_n * block_n],
            p.c_buffer,
        )
    tma.store_wait(pendings=0)


@gluon.jit
def _kernel(
    a_desc,
    b_desc,
    c_desc,
    num_buffers: gl.constexpr,
    num_warps: gl.constexpr,
):
    dtype: gl.constexpr = a_desc.dtype
    a_buffers = gl.allocate_shared_memory(
        dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout
    )
    b_buffers = gl.allocate_shared_memory(
        dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout
    )
    empty_barriers = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    ready_barriers = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    c_buffer = gl.allocate_shared_memory(
        dtype, c_desc.block_type.shape, c_desc.layout
    )
    for index in gl.static_range(num_buffers):
        mbarrier.init(empty_barriers.index(index), count=1)
        mbarrier.init(ready_barriers.index(index), count=1)

    args = PartitionArgs(
        a_desc,
        b_desc,
        c_desc,
        a_buffers,
        b_buffers,
        empty_barriers,
        ready_barriers,
        c_buffer,
        num_warps,
    )
    # Eight (or four) default warps own WGMMA and the epilogue. A dedicated
    # one-warp worker owns every TMA load; Hopper allocates it as a warpgroup.
    gl.warp_specialize(
        [(_compute_partition, (args,)), (_load_partition, (args,))],
        [1],
        [24],
    )


def _gluon_dtype(dtype: torch.dtype):
    return gl.float16 if dtype == torch.float16 else gl.bfloat16


def _config(m: int, n: int) -> tuple[int, int, int, int, int]:
    block_m = 64 if m <= 64 else 128
    block_n = 64 if n <= 64 else 128
    # 128x128x128 gives each compute thread 128 FP32 accumulator registers.
    # Three operand stages plus the epilogue tile occupy ~224 KiB, fitting the
    # H100 opt-in shared-memory limit while maximizing producer look-ahead.
    return block_m, block_n, 128, 3, 4


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not supports_tma(a.device) or torch.cuda.get_device_capability(a.device)[0] != 9:
        raise RuntimeError("this Gluon WGMMA kernel targets Hopper (SM90)")
    a, b, m, n, _k, original_n = aligned_inputs(a, b)
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    block_m, block_n, block_k, num_buffers, num_warps = _config(m, n)
    dtype = _gluon_dtype(a.dtype)

    a_shape = [block_m, block_k]
    b_shape = [block_k, block_n]
    c_shape = [block_m, block_n]
    a_desc = TensorDescriptor.from_tensor(
        a, a_shape, gl.NVMMASharedLayout.get_default_for(a_shape, dtype)
    )
    b_desc = TensorDescriptor.from_tensor(
        b, b_shape, gl.NVMMASharedLayout.get_default_for(b_shape, dtype)
    )
    c_desc = TensorDescriptor.from_tensor(
        c, c_shape, gl.NVMMASharedLayout.get_default_for(c_shape, dtype)
    )
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (min(sms, triton.cdiv(m, block_m) * triton.cdiv(n, block_n)),)
    _kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        num_buffers,
        num_warps=num_warps,
        maxnreg=232,
    )
    return restore_output(c, original_n)
