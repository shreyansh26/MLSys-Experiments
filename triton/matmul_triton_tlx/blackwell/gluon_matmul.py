"""Persistent Blackwell Gluon GEMM using TMA, TMEM, and tcgen05 MMA."""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
    tensor_memory_descriptor,
)
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate

from reference import aligned_inputs, restore_output, supports_tma, validate_inputs


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
        return gl.cdiv(self.tile_count - self.start_tile, gl.num_programs(0))

    @gluon.jit
    def get_tile(self, iteration):
        group_m: gl.constexpr = 8
        tile = self.start_tile + iteration * gl.num_programs(0)
        group = tile // self.tiles_per_group
        first_m = group * group_m
        actual_group_m = min(self.tiles_m - first_m, group_m)
        tile_in_group = tile % self.tiles_per_group
        return (
            first_m + tile_in_group % actual_group_m,
            tile_in_group // actual_group_m,
        )


@aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_buffers: gl.shared_memory_descriptor
    b_buffers: gl.shared_memory_descriptor
    load_empty: gl.shared_memory_descriptor
    load_ready: gl.shared_memory_descriptor
    accumulators: tensor_memory_descriptor
    acc_empty: gl.shared_memory_descriptor
    acc_ready: gl.shared_memory_descriptor
    epilogue_subtiles: gl.constexpr


@gluon.jit
def _load_partition(p):
    block_m: gl.constexpr = p.a_desc.block_type.shape[0]
    block_n: gl.constexpr = p.b_desc.block_type.shape[1]
    block_k: gl.constexpr = p.a_desc.block_type.shape[1]
    state = Counter.create(1, p.load_empty.shape[0])
    scheduler = GroupedPersistentScheduler.initialize(
        p.c_desc.shape[0], p.c_desc.shape[1], block_m, block_n
    )

    for tile_index in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(tile_index)
        offset_m = pid_m * block_m
        offset_n = pid_n * block_n
        for offset_k in range(0, p.a_desc.shape[1], block_k):
            empty = p.load_empty.index(state.index)
            ready = p.load_ready.index(state.index)
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
def _mma_partition(p):
    block_m: gl.constexpr = p.a_desc.block_type.shape[0]
    block_n: gl.constexpr = p.b_desc.block_type.shape[1]
    block_k: gl.constexpr = p.a_desc.block_type.shape[1]
    load_state = Counter.create(0, p.load_empty.shape[0])
    acc_state = Counter.create(1, p.acc_empty.shape[0])
    scheduler = GroupedPersistentScheduler.initialize(
        p.c_desc.shape[0], p.c_desc.shape[1], block_m, block_n
    )

    for _ in range(scheduler.get_num_tiles()):
        mbarrier.wait(p.acc_empty.index(acc_state.index), acc_state.phase)
        accumulator = p.accumulators.index(acc_state.index)
        use_accumulator = False
        for _ in range(0, p.a_desc.shape[1], block_k):
            mbarrier.wait(p.load_ready.index(load_state.index), load_state.phase)
            tcgen05_mma(
                p.a_buffers.index(load_state.index),
                p.b_buffers.index(load_state.index),
                accumulator,
                use_acc=use_accumulator,
            )
            tcgen05_commit(p.load_empty.index(load_state.index))
            load_state = load_state.next()
            use_accumulator = True
        tcgen05_commit(p.acc_ready.index(acc_state.index))
        acc_state = acc_state.next()


@gluon.jit
def _split_n(value, factor: gl.constexpr):
    split_count: gl.constexpr = factor.bit_length() - 1
    values = (value,)
    for _ in gl.static_range(split_count):
        next_values = ()
        for index in gl.static_range(len(values)):
            item = values[index]
            item = item.reshape(item.shape[0], 2, item.shape[1] // 2).permute(0, 2, 1)
            next_values += item.split()
        values = next_values
    return values


@gluon.jit
def _epilogue_partition(p):
    block_m: gl.constexpr = p.a_desc.block_type.shape[0]
    block_n: gl.constexpr = p.b_desc.block_type.shape[1]
    subtile_n: gl.constexpr = block_n // p.epilogue_subtiles
    state = Counter.create(0, p.acc_empty.shape[0])
    output_buffer = gl.allocate_shared_memory(
        p.c_desc.dtype, [block_m, subtile_n], p.c_desc.layout
    )
    scheduler = GroupedPersistentScheduler.initialize(
        p.c_desc.shape[0], p.c_desc.shape[1], block_m, block_n
    )

    for tile_index in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(tile_index)
        mbarrier.wait(p.acc_ready.index(state.index), state.phase)
        accumulator = p.accumulators.index(state.index).load()
        state = state.next()

        subtiles = _split_n(accumulator, p.epilogue_subtiles)
        for index in gl.static_range(len(subtiles)):
            value = subtiles[index].to(p.c_desc.dtype)
            tma.store_wait(pendings=0)
            output_buffer.store(value)
            if index == 0:
                mbarrier.arrive(p.acc_empty.index(state.index), count=1)
            fence_async_shared()
            tma.async_copy_shared_to_global(
                p.c_desc,
                [pid_m * block_m, pid_n * block_n + index * subtile_n],
                output_buffer,
            )
    tma.store_wait(pendings=0)


@gluon.jit
def _kernel(
    a_desc,
    b_desc,
    c_desc,
    num_buffers: gl.constexpr,
    epilogue_subtiles: gl.constexpr,
    num_warps: gl.constexpr,
):
    block_m: gl.constexpr = a_desc.block_type.shape[0]
    block_n: gl.constexpr = b_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype

    a_buffers = gl.allocate_shared_memory(
        dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout
    )
    b_buffers = gl.allocate_shared_memory(
        dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout
    )
    load_empty = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    load_ready = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for index in gl.static_range(num_buffers):
        mbarrier.init(load_empty.index(index), count=1)
        mbarrier.init(load_ready.index(index), count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout([block_m, block_n], col_stride=1)
    accumulators = allocate_tensor_memory(
        gl.float32, [2, block_m, block_n], tmem_layout
    )
    acc_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for index in gl.static_range(2):
        mbarrier.init(acc_empty.index(index), count=1)
        mbarrier.init(acc_ready.index(index), count=1)

    args = PartitionArgs(
        a_desc,
        b_desc,
        c_desc,
        a_buffers,
        b_buffers,
        load_empty,
        load_ready,
        accumulators,
        acc_empty,
        acc_ready,
        epilogue_subtiles,
    )
    gl.warp_specialize(
        [
            (_epilogue_partition, (args,)),
            (_load_partition, (args,)),
            (_mma_partition, (args,)),
        ],
        [1, 1],
        [24, 24],
    )


def _gluon_dtype(dtype: torch.dtype):
    return gl.float16 if dtype == torch.float16 else gl.bfloat16


def matmul_one_cta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not supports_tma(a.device) or torch.cuda.get_device_capability(a.device)[0] < 10:
        raise RuntimeError("this Gluon TCGen5 kernel requires Blackwell (SM100+)")
    a, b, m, n, _k, original_n = aligned_inputs(a, b)
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    dtype = _gluon_dtype(a.dtype)
    block_m, block_n, block_k = 128, 256, 64
    num_buffers, epilogue_subtiles, num_warps = 4, 4, 4

    a_shape = [block_m, block_k]
    b_shape = [block_k, block_n]
    c_shape = [block_m, block_n // epilogue_subtiles]
    a_desc = TensorDescriptor.from_tensor(
        a, a_shape, gl.NVMMASharedLayout.get_default_for(a_shape, dtype)
    )
    b_desc = TensorDescriptor.from_tensor(
        b, b_shape, gl.NVMMASharedLayout.get_default_for(b_shape, dtype)
    )
    c_desc = TensorDescriptor.from_tensor(
        c,
        c_shape,
        gl.NVMMASharedLayout.get_default_for([block_m, block_n], dtype),
    )

    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (min(sms, triton.cdiv(m, block_m) * triton.cdiv(n, block_n)),)
    _kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        num_buffers,
        epilogue_subtiles,
        num_warps=num_warps,
    )
    return restore_output(c, original_n)


def matmul_two_cta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from blackwell.gluon_matmul_2cta import matmul as implementation

    return implementation(a, b)


def use_two_ctas(m: int, n: int, k: int) -> bool:
    """Use clusters only when operand reuse amortizes their coordination cost."""
    return m >= 4096 and n >= 4096 and k >= 4096


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n, k = validate_inputs(a, b)
    if use_two_ctas(m, n, k):
        return matmul_two_cta(a, b)
    return matmul_one_cta(a, b)
