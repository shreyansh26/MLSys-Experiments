"""Two-CTA Blackwell Gluon GEMM with TMA multicast, CLC, and TCGen5.

The pipeline is adapted from Triton's Gluon multi-CTA tutorial and made
project-local here to support both input dtypes, alignment/cropping, explicit
benchmark providers, and shape-aware dispatch without tutorial side effects.
"""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    clc,
    tcgen05_commit,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
    tensor_memory_descriptor,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate

from reference import aligned_inputs, restore_output, supports_tma


BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 64
PIPELINE_STAGES = 6
ACCUMULATOR_STAGES = 2
EPILOGUE_N = 32
EPILOGUE_STAGES = 4
CGA_LAYOUT = ((1, 0),)


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


@gluon.constexpr_function
def _split_dim(cga_layout, dim):
    return 1 << sum(basis[dim] != 0 for basis in cga_layout)


def _operand_cga_layout(cga_layout, operand, two_ctas):
    """Broadcast each operand along GEMM K while preserving 2-CTA MMA bases."""
    if not cga_layout:
        return cga_layout

    def broadcast(basis):
        multiplier = 2 if two_ctas else 1
        return (basis[0], 0) if operand == 0 else (0, multiplier * basis[1])

    if not two_ctas:
        return tuple(map(broadcast, cga_layout))
    if cga_layout[0] != (1, 0):
        raise ValueError("the two-CTA kernel must split its output along M")
    first = (1, 0) if operand == 0 else (0, 1)
    return (first, *map(broadcast, cga_layout[1:]))


@gluon.jit
def _planar_snake(
    linear_tile,
    tiles_m,
    tiles_n,
    minor_dim: gl.constexpr,
    tile_width: gl.constexpr,
):
    major_size = tiles_n if minor_dim == 0 else tiles_m
    minor_size = tiles_m if minor_dim == 0 else tiles_n
    full_minor_tiles = minor_size // tile_width
    full_minor_size = full_minor_tiles * tile_width
    full_elements = full_minor_size * major_size
    minor_tile = linear_tile // (tile_width * major_size)

    full_minor = minor_tile * tile_width + linear_tile % tile_width
    full_major_index = (linear_tile // tile_width) % major_size
    full_major = gl.where(
        minor_tile % 2 == 0, full_major_index, major_size - 1 - full_major_index
    )

    partial_width = gl.where(minor_size - full_minor_size > 0, minor_size - full_minor_size, 1)
    partial_linear = linear_tile - full_elements
    partial_minor = minor_tile * tile_width + partial_linear % partial_width
    partial_major_index = (partial_linear // partial_width) % major_size
    partial_major = gl.where(
        minor_tile % 2 == 0,
        partial_major_index,
        major_size - 1 - partial_major_index,
    )

    in_full_tile = linear_tile < full_elements
    minor = gl.where(in_full_tile, full_minor, partial_minor)
    major = gl.where(in_full_tile, full_major, partial_major)
    if minor_dim == 0:
        return minor, major
    return major, minor


@aggregate
class ClcScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    pid_m: gl.tensor
    pid_n: gl.tensor
    tiles_m: gl.tensor
    tiles_n: gl.tensor
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    MINOR_DIM: gl.constexpr
    TILE_WIDTH: gl.constexpr
    results: gl.shared_memory_descriptor
    result_ready: gl.shared_memory_descriptor
    planar_tiles: gl.shared_memory_descriptor
    planar_ready: gl.shared_memory_descriptor
    consumed: gl.shared_memory_descriptor
    state: Counter
    consumed_state: Counter

    @gluon.jit
    def initialize(
        m,
        n,
        tile_m: gl.constexpr,
        tile_n: gl.constexpr,
        minor_dim: gl.constexpr,
        tile_width: gl.constexpr,
        results,
        result_ready,
        planar_tiles,
        planar_ready,
        consumed,
    ):
        tile_id = gl.program_id(0)
        tiles_m = gl.cdiv(m, tile_m)
        tiles_n = gl.cdiv(n, tile_n)
        pid_m, pid_n = _planar_snake(
            tile_id, tiles_m, tiles_n, minor_dim, tile_width
        )
        return ClcScheduler(
            gl.to_tensor(True),
            tile_id,
            pid_m,
            pid_n,
            tiles_m,
            tiles_n,
            tile_m,
            tile_n,
            minor_dim,
            tile_width,
            results,
            result_ready,
            planar_tiles,
            planar_ready,
            consumed,
            Counter.create(0, result_ready.shape[0]),
            Counter.create(0, result_ready.shape[0]),
        )

    @gluon.jit
    def offsets(self):
        return self.pid_m * self.TILE_M, self.pid_n * self.TILE_N

    @gluon.must_use_result
    @gluon.jit
    def next(self, iteration):
        consumed_state = self.consumed_state
        if iteration > 0:
            mbarrier.arrive(self.consumed.index(consumed_state.index))
            consumed_state = consumed_state.next()

        state = self.state
        mbarrier.wait(self.result_ready.index(state.index), state.phase)
        result = clc.load_result(self.results.index(state.index))
        mbarrier.wait(self.planar_ready.index(state.index), state.phase)
        scalar_layout: gl.constexpr = gl.BlockedLayout(
            [1], [32], [gl.num_warps()], [0], [[0]]
        )
        packed = self.planar_tiles.index(state.index).load(scalar_layout).reshape([])
        pid_m = ((packed >> 32) & 0xFFFFFFFF).to(gl.int32)
        pid_n = (packed & 0xFFFFFFFF).to(gl.int32)
        has_work = result.is_canceled()
        tile_id = self.tile_id
        if has_work:
            tile_id = result.program_id(0)
        return ClcScheduler(
            has_work,
            tile_id,
            pid_m,
            pid_n,
            self.tiles_m,
            self.tiles_n,
            self.TILE_M,
            self.TILE_N,
            self.MINOR_DIM,
            self.TILE_WIDTH,
            self.results,
            self.result_ready,
            self.planar_tiles,
            self.planar_ready,
            self.consumed,
            state.next(),
            consumed_state,
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
    clc_results: gl.shared_memory_descriptor
    clc_ready: gl.shared_memory_descriptor
    planar_tiles: gl.shared_memory_descriptor
    planar_ready: gl.shared_memory_descriptor
    clc_consumed: gl.shared_memory_descriptor
    MINOR_DIM: gl.constexpr
    TILE_WIDTH: gl.constexpr
    EPILOGUE_STAGES: gl.constexpr

    @gluon.jit
    def scheduler(self):
        return ClcScheduler.initialize(
            self.c_desc.shape[0],
            self.c_desc.shape[1],
            self.a_desc.block_shape[0],
            self.b_desc.block_shape[1],
            self.MINOR_DIM,
            self.TILE_WIDTH,
            self.clc_results,
            self.clc_ready,
            self.planar_tiles,
            self.planar_ready,
            self.clc_consumed,
        )


@gluon.jit
def _clc_partition(p):
    tile_m: gl.constexpr = p.a_desc.block_shape[0]
    tile_n: gl.constexpr = p.b_desc.block_shape[1]
    tiles_m = gl.cdiv(p.c_desc.shape[0], tile_m)
    tiles_n = gl.cdiv(p.c_desc.shape[1], tile_n)
    state = Counter.create(0, p.clc_ready.shape[0])
    consumed_state = Counter.create(1, p.clc_ready.shape[0])
    iteration = 0
    has_work = gl.to_tensor(True)
    while has_work:
        mbarrier.wait(
            p.clc_consumed.index(consumed_state.index),
            consumed_state.phase,
            pred=iteration >= p.clc_ready.shape[0],
        )
        ready = p.clc_ready.index(state.index)
        result_buffer = p.clc_results.index(state.index)
        mbarrier.expect(ready, 16)
        clc.try_cancel(result_buffer, ready, multicast=True)
        mbarrier.wait(ready, state.phase)
        result = clc.load_result(result_buffer)
        has_work = result.is_canceled()
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            tile_id = result.program_id(0)
            pid_m, pid_n = _planar_snake(
                tile_id, tiles_m, tiles_n, p.MINOR_DIM, p.TILE_WIDTH
            )
        packed = (pid_m.to(gl.int64) << 32) | (pid_n.to(gl.int64) & 0xFFFFFFFF)
        scalar_layout: gl.constexpr = gl.BlockedLayout(
            [1], [32], [gl.num_warps()], [0], [[0]]
        )
        p.planar_tiles.index(state.index).store(
            gl.full([1], packed, gl.int64, layout=scalar_layout)
        )
        mbarrier.arrive(p.planar_ready.index(state.index))
        state = state.next()
        consumed_state = consumed_state.next()
        iteration += 1


@gluon.jit
def _load_partition(p):
    block_k: gl.constexpr = p.a_desc.block_shape[1]
    state = Counter.create(1, p.load_ready.shape[0])
    scheduler = p.scheduler()
    iteration = 0
    while scheduler.has_work:
        offset_m, offset_n = scheduler.offsets()
        for k in range(0, p.a_desc.shape[1], block_k):
            pred = iteration > 0 or k >= block_k * p.load_ready.shape[0]
            mbarrier.wait(p.load_empty.index(state.index), state.phase, pred=pred)
            ready = p.load_ready.index(state.index)
            mbarrier.expect(ready, p.a_desc.nbytes_per_cta + p.b_desc.nbytes_per_cta)
            tma.async_copy_global_to_shared(
                p.a_desc,
                [offset_m, k],
                ready,
                p.a_buffers.index(state.index),
                multicast=True,
            )
            tma.async_copy_global_to_shared(
                p.b_desc,
                [k, offset_n],
                ready,
                p.b_buffers.index(state.index),
                multicast=True,
            )
            state = state.next()
        scheduler = scheduler.next(iteration)
        iteration += 1


@gluon.jit
def _mma_partition(p):
    block_k: gl.constexpr = p.a_desc.block_shape[1]
    load_state = Counter.create(0, p.load_ready.shape[0])
    acc_state = Counter.create(1, p.acc_ready.shape[0])
    scheduler = p.scheduler()
    iteration = 0
    while scheduler.has_work:
        mbarrier.wait(
            p.acc_empty.index(acc_state.index),
            acc_state.phase,
            pred=iteration >= p.acc_ready.shape[0],
        )
        accumulator = p.accumulators.index(acc_state.index)
        use_acc = False
        for _ in range(0, p.a_desc.shape[1], block_k):
            mbarrier.wait(p.load_ready.index(load_state.index), load_state.phase)
            tcgen05_mma(
                p.a_buffers.index(load_state.index),
                p.b_buffers.index(load_state.index),
                accumulator,
                use_acc=use_acc,
                multicast=True,
                mbarriers=[p.load_empty.index(load_state.index)],
            )
            load_state = load_state.next()
            use_acc = True
        tcgen05_commit(
            p.acc_ready.index(acc_state.index),
            descs=[p.a_buffers.index(0), p.b_buffers.index(0)],
        )
        acc_state = acc_state.next()
        scheduler = scheduler.next(iteration)
        iteration += 1


@gluon.jit
def _epilogue_partition(p):
    tile_m: gl.constexpr = p.a_desc.block_shape[0]
    tile_n: gl.constexpr = p.b_desc.block_shape[1]
    epilogue_n: gl.constexpr = p.c_desc.block_shape[1]
    subtile_count: gl.constexpr = tile_n // epilogue_n
    buffers = gl.allocate_shared_memory(
        p.c_desc.dtype,
        [p.EPILOGUE_STAGES, tile_m, epilogue_n],
        p.c_desc.layout,
    )
    acc_state = Counter.create(0, p.acc_ready.shape[0])
    store_state = Counter.create(0, p.EPILOGUE_STAGES)
    scheduler = p.scheduler()
    iteration = 0
    while scheduler.has_work:
        offset_m, offset_n = scheduler.offsets()
        mbarrier.wait(p.acc_ready.index(acc_state.index), acc_state.phase)
        accumulator = p.accumulators.index(acc_state.index)
        for subtile in gl.static_range(subtile_count):
            output_buffer = buffers.index(store_state.index)
            output = accumulator.slice(epilogue_n * subtile, epilogue_n).load()
            tma.store_wait(pendings=p.EPILOGUE_STAGES - 1)
            output_buffer.store(output.to(p.c_desc.dtype))
            tma.async_copy_shared_to_global(
                p.c_desc,
                [offset_m, offset_n + epilogue_n * subtile],
                output_buffer,
            )
            store_state = store_state.next()
        mbarrier.arrive(p.acc_empty.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.next(iteration)
        iteration += 1


@gluon.jit
def _kernel(
    a_desc,
    b_desc,
    c_desc,
    stages: gl.constexpr,
    acc_stages: gl.constexpr,
    cga_layout: gl.constexpr,
    minor_dim: gl.constexpr,
    tile_width: gl.constexpr,
    epilogue_stages: gl.constexpr,
    num_warps: gl.constexpr,
):
    dtype: gl.constexpr = a_desc.dtype
    two_ctas: gl.constexpr = gl.num_ctas() == 2
    a_buffers = gl.allocate_shared_memory(
        dtype, [stages] + a_desc.block_shape, a_desc.layout
    )
    b_buffers = gl.allocate_shared_memory(
        dtype, [stages] + b_desc.block_shape, b_desc.layout
    )
    mma_arrivals: gl.constexpr = tcgen05_mma_barrier_count(
        [a_buffers.index(0), b_buffers.index(0)], multicast=True
    )
    load_empty = mbarrier.allocate_mbarrier(batch=stages)
    load_ready = mbarrier.allocate_mbarrier(batch=stages, two_ctas=two_ctas)
    for index in gl.static_range(stages):
        mbarrier.init(load_empty.index(index), count=mma_arrivals)
        mbarrier.init(load_ready.index(index), count=1)

    tmem_layout: gl.constexpr = TensorMemoryLayout(
        [
            a_desc.block_shape[0] // _split_dim(cga_layout, 0),
            b_desc.block_shape[1] // _split_dim(cga_layout, 1),
        ],
        col_stride=1,
        cga_layout=cga_layout,
        two_ctas=two_ctas,
    )
    accumulators = allocate_tensor_memory(
        gl.float32,
        [acc_stages, a_desc.block_shape[0], b_desc.block_shape[1]],
        tmem_layout,
    )
    acc_empty = mbarrier.allocate_mbarrier(batch=acc_stages, two_ctas=two_ctas)
    acc_ready = mbarrier.allocate_mbarrier(batch=acc_stages)
    for index in gl.static_range(acc_stages):
        mbarrier.init(acc_empty.index(index), count=1)
        mbarrier.init(acc_ready.index(index), count=mma_arrivals)

    clc_ready = mbarrier.allocate_mbarrier(batch=acc_stages)
    planar_ready = mbarrier.allocate_mbarrier(batch=acc_stages)
    clc_consumed = mbarrier.allocate_mbarrier(batch=acc_stages, two_ctas=two_ctas)
    for index in gl.static_range(acc_stages):
        mbarrier.init(clc_ready.index(index), count=1)
        mbarrier.init(planar_ready.index(index), count=1)
        mbarrier.init(clc_consumed.index(index), count=3)

    scalar_layout: gl.constexpr = gl.SwizzledSharedLayout(
        1, 1, 1, [0], cga_layout=[[0]]
    )
    clc_results = gl.allocate_shared_memory(
        gl.int64, [acc_stages, 2], scalar_layout
    )
    planar_tiles = gl.allocate_shared_memory(
        gl.int64, [acc_stages, 1], scalar_layout
    )
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
        clc_results,
        clc_ready,
        planar_tiles,
        planar_ready,
        clc_consumed,
        minor_dim,
        tile_width,
        epilogue_stages,
    )
    gl.warp_specialize(
        [
            (_epilogue_partition, (args,)),
            (_load_partition, (args,)),
            (_mma_partition, (args,)),
            (_clc_partition, (args,)),
        ],
        [1, 1, 1],
        [24, 24, 24],
    )


def _gluon_dtype(dtype: torch.dtype):
    return gl.float16 if dtype == torch.float16 else gl.bfloat16


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not supports_tma(a.device) or torch.cuda.get_device_capability(a.device)[0] < 10:
        raise RuntimeError("this two-CTA Gluon kernel requires Blackwell (SM100+)")
    a, b, m, n, _k, original_n = aligned_inputs(a, b)
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    dtype = _gluon_dtype(a.dtype)
    tile_m = BLOCK_M * _split_dim(CGA_LAYOUT, 0)
    a_layout = gl.NVMMASharedLayout.get_default_for(
        [tile_m, BLOCK_K],
        dtype,
        cga_layout=_operand_cga_layout(CGA_LAYOUT, 0, True),
    )
    b_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_K, BLOCK_N],
        dtype,
        cga_layout=_operand_cga_layout(CGA_LAYOUT, 1, True),
    )
    c_layout = gl.NVMMASharedLayout.get_default_for(
        [tile_m, EPILOGUE_N], dtype, cga_layout=CGA_LAYOUT
    )
    a_desc = TensorDescriptor.from_tensor(a, [tile_m, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(output, [tile_m, EPILOGUE_N], c_layout)

    grid = (triton.cdiv(m, tile_m) * triton.cdiv(n, BLOCK_N),)
    _kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        PIPELINE_STAGES,
        ACCUMULATOR_STAGES,
        CGA_LAYOUT,
        0,
        16,
        EPILOGUE_STAGES,
        num_warps=4,
        num_ctas=2,
    )
    return restore_output(output, original_n)
