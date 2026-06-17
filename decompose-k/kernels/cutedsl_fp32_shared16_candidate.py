"""CuteDSL shared-memory FP32 split-K candidate with staged cp.async tiles."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import dsl_user_op

from .cutedsl_splitk_gemm_candidate import _compiled_vec4_reducer
from kernels.decompose_k_triton_kernel import KernelConfig, _check_inputs


K_SPLITS = (16, 32, 64, 128, 256)
_DOT_CACHE = {}


@dsl_user_op
def _cp_async_cg_16(dst_ptr, src_ptr, *, loc=None, ip=None) -> None:
    dst_i32 = dst_ptr.toint(loc=loc, ip=ip).ir_value()
    src_i64 = src_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [dst_i32, src_i64],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.kernel
def _shared_tile_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    partials: cute.Tensor,
    block_m: cutlass.Constexpr,
    block_n: cutlass.Constexpr,
    block_k: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    thread_m: cutlass.Constexpr,
    allow_edges: cutlass.Constexpr,
):
    pid, split_id, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    n_tiles = (b.shape[1] + block_n - 1) // block_n
    pid_m = pid // n_tiles
    pid_n = pid - pid_m * n_tiles
    m_base = pid_m * block_m
    n_base = pid_n * block_n
    k_per_split = a.shape[1] // partials.shape[0]
    k_start = split_id * k_per_split

    sA_layout = cute.make_layout(
        (block_m, block_k, 2), stride=(block_k, 1, block_m * block_k)
    )
    sB_layout = cute.make_layout(
        (block_k, block_n, 2), stride=(block_n, 1, block_k * block_n)
    )
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cute.Float32, sA_layout, 16)
    sB = smem.allocate_tensor(cute.Float32, sB_layout, 16)

    vec_type = ir.VectorType.get([4], cutlass.Float32.mlir_type)
    n_vecs = block_n // 4
    row_groups = block_m // thread_m
    output_vecs = row_groups * n_vecs
    out_vec_idx = tidx
    local_m_group = out_vec_idx // n_vecs
    local_m_base = local_m_group * thread_m
    local_n = (out_vec_idx - local_m_group * n_vecs) * 4
    m = m_base + local_m_base
    n = n_base + local_n

    acc00 = cutlass.Float32(0.0)
    acc01 = cutlass.Float32(0.0)
    acc02 = cutlass.Float32(0.0)
    acc03 = cutlass.Float32(0.0)
    acc10 = cutlass.Float32(0.0)
    acc11 = cutlass.Float32(0.0)
    acc12 = cutlass.Float32(0.0)
    acc13 = cutlass.Float32(0.0)
    acc20 = cutlass.Float32(0.0)
    acc21 = cutlass.Float32(0.0)
    acc22 = cutlass.Float32(0.0)
    acc23 = cutlass.Float32(0.0)
    acc30 = cutlass.Float32(0.0)
    acc31 = cutlass.Float32(0.0)
    acc32 = cutlass.Float32(0.0)
    acc33 = cutlass.Float32(0.0)
    k_vecs = block_k // 4

    for idx in range(tidx, block_m * k_vecs, threads_per_block):
        local_m = idx // k_vecs
        local_k = (idx - local_m * k_vecs) * 4
        load_m = m_base + local_m
        src_ptr = a.iterator + load_m * a.shape[1] + k_start + local_k
        dst_ptr = sA.iterator + local_m * block_k + local_k
        if cutlass.const_expr(allow_edges):
            if load_m < a.shape[0]:
                vals = cute.arch.load(src_ptr, vec_type, cop="cg")
                cute.arch.store(dst_ptr, vals, ss="cta")
            else:
                sA[local_m, local_k + 0, 0] = cutlass.Float32(0.0)
                sA[local_m, local_k + 1, 0] = cutlass.Float32(0.0)
                sA[local_m, local_k + 2, 0] = cutlass.Float32(0.0)
                sA[local_m, local_k + 3, 0] = cutlass.Float32(0.0)
        else:
            _cp_async_cg_16(dst_ptr, src_ptr)

    for idx in range(tidx, block_k * n_vecs, threads_per_block):
        local_k = idx // n_vecs
        local_n_load = (idx - local_k * n_vecs) * 4
        load_n = n_base + local_n_load
        src_ptr = b.iterator + (k_start + local_k) * b.shape[1] + load_n
        dst_ptr = sB.iterator + local_k * block_n + local_n_load
        if cutlass.const_expr(allow_edges):
            if load_n + 3 < b.shape[1]:
                vals = cute.arch.load(src_ptr, vec_type, cop="cg")
                cute.arch.store(dst_ptr, vals, ss="cta")
            else:
                sB[local_k, local_n_load + 0, 0] = cutlass.Float32(0.0)
                sB[local_k, local_n_load + 1, 0] = cutlass.Float32(0.0)
                sB[local_k, local_n_load + 2, 0] = cutlass.Float32(0.0)
                sB[local_k, local_n_load + 3, 0] = cutlass.Float32(0.0)
        else:
            _cp_async_cg_16(dst_ptr, src_ptr)

    if cutlass.const_expr(allow_edges):
        cute.arch.sync_threads()
    else:
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

    for k_block in cutlass.range_constexpr(0, k_per_split, block_k):
        read_stage = (k_block // block_k) % 2
        next_k_block = k_block + block_k
        if cutlass.const_expr(next_k_block < k_per_split):
            write_stage = 1 - read_stage
            write_offset_a = write_stage * block_m * block_k
            write_offset_b = write_stage * block_k * block_n
            for idx in range(tidx, block_m * k_vecs, threads_per_block):
                local_m = idx // k_vecs
                local_k = (idx - local_m * k_vecs) * 4
                load_m = m_base + local_m
                src_ptr = (
                    a.iterator + load_m * a.shape[1] + k_start + next_k_block + local_k
                )
                dst_ptr = sA.iterator + write_offset_a + local_m * block_k + local_k
                if cutlass.const_expr(allow_edges):
                    if load_m < a.shape[0]:
                        vals = cute.arch.load(src_ptr, vec_type, cop="cg")
                        cute.arch.store(dst_ptr, vals, ss="cta")
                    else:
                        sA[local_m, local_k + 0, write_stage] = cutlass.Float32(0.0)
                        sA[local_m, local_k + 1, write_stage] = cutlass.Float32(0.0)
                        sA[local_m, local_k + 2, write_stage] = cutlass.Float32(0.0)
                        sA[local_m, local_k + 3, write_stage] = cutlass.Float32(0.0)
                else:
                    _cp_async_cg_16(dst_ptr, src_ptr)

            for idx in range(tidx, block_k * n_vecs, threads_per_block):
                local_k = idx // n_vecs
                local_n_load = (idx - local_k * n_vecs) * 4
                load_n = n_base + local_n_load
                src_ptr = (
                    b.iterator
                    + (k_start + next_k_block + local_k) * b.shape[1]
                    + load_n
                )
                dst_ptr = sB.iterator + write_offset_b + local_k * block_n + local_n_load
                if cutlass.const_expr(allow_edges):
                    if load_n + 3 < b.shape[1]:
                        vals = cute.arch.load(src_ptr, vec_type, cop="cg")
                        cute.arch.store(dst_ptr, vals, ss="cta")
                    else:
                        sB[local_k, local_n_load + 0, write_stage] = cutlass.Float32(0.0)
                        sB[local_k, local_n_load + 1, write_stage] = cutlass.Float32(0.0)
                        sB[local_k, local_n_load + 2, write_stage] = cutlass.Float32(0.0)
                        sB[local_k, local_n_load + 3, write_stage] = cutlass.Float32(0.0)
                else:
                    _cp_async_cg_16(dst_ptr, src_ptr)

            if cutlass.const_expr(not allow_edges):
                cute.arch.cp_async_commit_group()

        if out_vec_idx < output_vecs and m < a.shape[0] and n + 3 < b.shape[1]:
            read_offset_a = read_stage * block_m * block_k
            read_offset_b = read_stage * block_k * block_n
            for kk in cutlass.range_constexpr(0, block_k):
                b_vals = cute.arch.load(
                    sB.iterator + read_offset_b + kk * block_n + local_n,
                    vec_type,
                    ss="cta",
                )
                b0 = cutlass.Float32(vector.extract(b_vals, [], [0]))
                b1 = cutlass.Float32(vector.extract(b_vals, [], [1]))
                b2 = cutlass.Float32(vector.extract(b_vals, [], [2]))
                b3 = cutlass.Float32(vector.extract(b_vals, [], [3]))
                a0 = cute.arch.load(
                    sA.iterator + read_offset_a + local_m_base * block_k + kk,
                    cutlass.Float32,
                    ss="cta",
                )
                acc00 += a0 * b0
                acc01 += a0 * b1
                acc02 += a0 * b2
                acc03 += a0 * b3
                if cutlass.const_expr(thread_m >= 2):
                    a1 = cute.arch.load(
                        sA.iterator + read_offset_a + (local_m_base + 1) * block_k + kk,
                        cutlass.Float32,
                        ss="cta",
                    )
                    acc10 += a1 * b0
                    acc11 += a1 * b1
                    acc12 += a1 * b2
                    acc13 += a1 * b3
                if cutlass.const_expr(thread_m == 4):
                    a2 = cute.arch.load(
                        sA.iterator + read_offset_a + (local_m_base + 2) * block_k + kk,
                        cutlass.Float32,
                        ss="cta",
                    )
                    a3 = cute.arch.load(
                        sA.iterator + read_offset_a + (local_m_base + 3) * block_k + kk,
                        cutlass.Float32,
                        ss="cta",
                    )
                    acc20 += a2 * b0
                    acc21 += a2 * b1
                    acc22 += a2 * b2
                    acc23 += a2 * b3
                    acc30 += a3 * b0
                    acc31 += a3 * b1
                    acc32 += a3 * b2
                    acc33 += a3 * b3

        if cutlass.const_expr(next_k_block < k_per_split):
            if cutlass.const_expr(allow_edges):
                cute.arch.sync_threads()
            else:
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

    if out_vec_idx < output_vecs and m < a.shape[0] and n + 3 < b.shape[1]:
        partials[split_id, m + 0, n + 0] = acc00
        partials[split_id, m + 0, n + 1] = acc01
        partials[split_id, m + 0, n + 2] = acc02
        partials[split_id, m + 0, n + 3] = acc03
        if cutlass.const_expr(thread_m >= 2):
            partials[split_id, m + 1, n + 0] = acc10
            partials[split_id, m + 1, n + 1] = acc11
            partials[split_id, m + 1, n + 2] = acc12
            partials[split_id, m + 1, n + 3] = acc13
        if cutlass.const_expr(thread_m == 4):
            partials[split_id, m + 2, n + 0] = acc20
            partials[split_id, m + 2, n + 1] = acc21
            partials[split_id, m + 2, n + 2] = acc22
            partials[split_id, m + 2, n + 3] = acc23
            partials[split_id, m + 3, n + 0] = acc30
            partials[split_id, m + 3, n + 1] = acc31
            partials[split_id, m + 3, n + 2] = acc32
            partials[split_id, m + 3, n + 3] = acc33


@cute.jit
def _shared_tile_launch(
    a: cute.Tensor,
    b: cute.Tensor,
    partials: cute.Tensor,
    block_m: cutlass.Constexpr,
    block_n: cutlass.Constexpr,
    block_k: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
    thread_m: cutlass.Constexpr,
    allow_edges: cutlass.Constexpr,
):
    sA_layout = cute.make_layout(
        (block_m, block_k, 2), stride=(block_k, 1, block_m * block_k)
    )
    sB_layout = cute.make_layout(
        (block_k, block_n, 2), stride=(block_n, 1, block_k * block_n)
    )
    smem_size = cute.size_in_bytes(cute.Float32, sA_layout) + cute.size_in_bytes(
        cute.Float32, sB_layout
    )
    n_tiles = (b.shape[1] + block_n - 1) // block_n
    m_tiles = (a.shape[0] + block_m - 1) // block_m
    _shared_tile_kernel(
        a,
        b,
        partials,
        block_m,
        block_n,
        block_k,
        threads_per_block,
        thread_m,
        allow_edges,
    ).launch(
        grid=(m_tiles * n_tiles, partials.shape[0], 1),
        block=(threads_per_block, 1, 1),
        smem=smem_size,
    )


def partials_dtype(module_suite_name: str, input_dtype: torch.dtype) -> torch.dtype:
    del module_suite_name, input_dtype
    return torch.float32


def inductor_like_splits(m: int, n: int, k: int, limit: int) -> list[int]:
    del limit
    max_split = min(k // m, k // n)
    splits = [
        split
        for split in K_SPLITS
        if split <= max_split
        and k % split == 0
        and 32 <= k // split <= 256
        and (k // split) % 32 == 0
    ]
    for k_per_split in (32, 64, 96, 128, 160, 192, 224, 256):
        if k % k_per_split == 0:
            split = k // k_per_split
            if split <= max_split and split not in splits:
                splits.append(split)
    return [
        split
        for split in splits
        if 32 <= k // split <= 256 and (k // split) % 32 == 0
    ]


def candidate_configs(split_values: list[int]) -> list[KernelConfig]:
    configs = []
    for split_k in split_values:
        for block_m, block_n, block_k, num_warps, thread_m in (
            (16, 16, 32, 4, 1),
            (16, 16, 64, 4, 1),
            (16, 16, 32, 8, 1),
            (16, 16, 64, 8, 1),
            (16, 32, 32, 8, 2),
            (16, 32, 64, 8, 2),
            (16, 32, 32, 4, 4),
            (32, 32, 32, 8, 2),
            (32, 32, 64, 8, 2),
            (32, 32, 32, 8, 4),
            (32, 32, 64, 8, 4),
            (32, 64, 32, 8, 4),
            (32, 64, 64, 8, 4),
            (48, 48, 32, 8, 4),
            (48, 48, 64, 8, 4),
            (64, 64, 32, 8, 4),
            (64, 64, 64, 8, 4),
        ):
            configs.append(
                KernelConfig(
                    split_k, block_m, block_n, block_k, 1, num_warps, thread_m
                )
            )
    return configs


def _compiled_dot(
    split_k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    thread_m: int = 1,
    allow_edges: bool = False,
):
    key = (
        split_k,
        block_m,
        block_n,
        block_k,
        num_warps,
        dtype,
        m,
        n,
        k,
        thread_m,
        allow_edges,
        "shared_tile_tvm_ffi",
    )
    compiled = _DOT_CACHE.get(key)
    if compiled is None:
        if dtype is not torch.float32:
            raise TypeError(f"unsupported shared16 dtype: {dtype}")
        a = cute.runtime.make_fake_tensor(
            cute.Float32,
            (m, k),
            (k, 1),
            assumed_align=16,
        )
        b = cute.runtime.make_fake_tensor(
            cute.Float32,
            (k, n),
            (n, 1),
            assumed_align=16,
        )
        partials = cute.runtime.make_fake_tensor(
            cute.Float32,
            (split_k, m, n),
            (m * n, n, 1),
            assumed_align=16,
        )
        compiled = cute.compile(
            _shared_tile_launch,
            a,
            b,
            partials,
            block_m,
            block_n,
            block_k,
            num_warps * 32,
            thread_m,
            allow_edges,
            options="--enable-tvm-ffi",
        )
        _DOT_CACHE[key] = compiled
    return compiled


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
    if a.dtype is not torch.float32 or b.dtype is not torch.float32:
        raise TypeError("CuteDSL shared-tile candidate requires FP32 inputs")
    if partials.dtype is not torch.float32:
        raise TypeError("CuteDSL shared-tile candidate requires FP32 partials")
    if (
        config.block_k not in (32, 64, 128, 256)
        or config.num_stages not in (1, 2, 4)
        or config.block_m % config.num_stages != 0
        or (a.shape[1] // config.split_k) % config.block_k != 0
        or config.block_k > (a.shape[1] // config.split_k)
        or config.block_m > 64
        or config.block_n > 64
        or (config.block_m == 48 and a.shape[0] % 48 != 0)
        or (config.block_n == 48 and b.shape[1] % 48 != 0)
        or (min(a.shape[0], b.shape[1]) < 48 and (config.block_m > 32 or config.block_n > 32))
    ):
        raise ValueError("unsupported CuteDSL shared-tile candidate tile")
    allow_edges = (
        a.shape[0] % config.block_m != 0 or b.shape[1] % config.block_n != 0
    )

    dot = _compiled_dot(
        config.split_k,
        config.block_m,
        config.block_n,
        config.block_k,
        config.num_warps,
        a.dtype,
        a.shape[0],
        b.shape[1],
        a.shape[1],
        config.num_stages,
        allow_edges,
    )
    dot(a, b, partials)

    reducer = _compiled_vec4_reducer(
        config.split_k, c.dtype, fuse_relu, c.shape[0], c.shape[1]
    )
    reducer(partials, c.reshape(-1))
    return c


def decompose_k_matmul_out(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    partials: torch.Tensor,
    config: KernelConfig,
) -> torch.Tensor:
    return decompose_k_relu_out(a, b, c, partials, config, fuse_relu=False)
