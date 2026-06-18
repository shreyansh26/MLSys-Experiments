"""Full CuteDSL split-K GEMM candidate for small-M/N, large-K matmuls.

This module maps split-K onto the CuteDSL GEMM tutorial's batch dimension:
A is viewed as (M, K_per_split, split_k), B as (N, K_per_split, split_k),
and partials as (M, N, split_k).  The final split reduction reuses the
CuteDSL reducer candidate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import vector
from cutlass.cute.runtime import from_dlpack

from kernels.decompose_k_triton_kernel import (
    KernelConfig,
    _check_inputs,
    inductor_like_splits as _baseline_splits,
)


K_SPLITS = (2, 4, 8, 16, 32, 64, 128, 256)
_GEMM_CACHE = {}
_TENSOR_CACHE = {}
_VEC4_REDUCER_CACHE = {}


@cute.kernel
def _reduce_vec4_kernel(
    partials: cute.Tensor,
    c: cute.Tensor,
    fuse_relu: cutlass.Constexpr,
):
    bid, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(partials.shape[0] <= 64):
        xblock = 32
        vec_groups = 8
        lanes_per_vec = 4
    elif cutlass.const_expr(partials.shape[0] <= 128):
        xblock = 16
        vec_groups = 4
        lanes_per_vec = 8
    else:
        xblock = 8
        vec_groups = 2
        lanes_per_vec = 16

    split_lane = tidx // vec_groups
    vec_group = tidx - split_lane * vec_groups
    x_base = bid * xblock + vec_group * 4
    xnumel = c.shape[0]

    acc0 = cutlass.Float32(0.0)
    acc1 = cutlass.Float32(0.0)
    acc2 = cutlass.Float32(0.0)
    acc3 = cutlass.Float32(0.0)
    vec_type = ir.VectorType.get([4], cutlass.Float32.mlir_type)

    if x_base + 3 < xnumel:
        for split_base in cutlass.range_constexpr(
            0, partials.shape[0], lanes_per_vec
        ):
            split_id = split_base + split_lane
            if split_id < partials.shape[0]:
                ptr = partials.iterator + split_id * xnumel + x_base
                vals = cute.arch.load(ptr, vec_type, cop="cg")
                acc0 += cutlass.Float32(vector.extract(vals, [], [0]))
                acc1 += cutlass.Float32(vector.extract(vals, [], [1]))
                acc2 += cutlass.Float32(vector.extract(vals, [], [2]))
                acc3 += cutlass.Float32(vector.extract(vals, [], [3]))

    acc0 += cute.arch.shuffle_sync_bfly(acc0, 16)
    acc1 += cute.arch.shuffle_sync_bfly(acc1, 16)
    acc2 += cute.arch.shuffle_sync_bfly(acc2, 16)
    acc3 += cute.arch.shuffle_sync_bfly(acc3, 16)
    acc0 += cute.arch.shuffle_sync_bfly(acc0, 8)
    acc1 += cute.arch.shuffle_sync_bfly(acc1, 8)
    acc2 += cute.arch.shuffle_sync_bfly(acc2, 8)
    acc3 += cute.arch.shuffle_sync_bfly(acc3, 8)
    if cutlass.const_expr(vec_groups <= 4):
        acc0 += cute.arch.shuffle_sync_bfly(acc0, 4)
        acc1 += cute.arch.shuffle_sync_bfly(acc1, 4)
        acc2 += cute.arch.shuffle_sync_bfly(acc2, 4)
        acc3 += cute.arch.shuffle_sync_bfly(acc3, 4)
    if cutlass.const_expr(vec_groups <= 2):
        acc0 += cute.arch.shuffle_sync_bfly(acc0, 2)
        acc1 += cute.arch.shuffle_sync_bfly(acc1, 2)
        acc2 += cute.arch.shuffle_sync_bfly(acc2, 2)
        acc3 += cute.arch.shuffle_sync_bfly(acc3, 2)

    if split_lane == 0 and x_base < xnumel:
        if fuse_relu:
            zero = cutlass.Float32(0.0)
            acc0 = cutlass.max(acc0, zero)
            acc1 = cutlass.max(acc1, zero)
            acc2 = cutlass.max(acc2, zero)
            acc3 = cutlass.max(acc3, zero)
        c[x_base + 0] = acc0.to(c.element_type)
        c[x_base + 1] = acc1.to(c.element_type)
        c[x_base + 2] = acc2.to(c.element_type)
        c[x_base + 3] = acc3.to(c.element_type)


@cute.jit
def _reduce_vec4_launch(
    partials: cute.Tensor,
    c: cute.Tensor,
    fuse_relu: cutlass.Constexpr,
):
    if cutlass.const_expr(partials.shape[0] <= 64):
        xblock = 32
    elif cutlass.const_expr(partials.shape[0] <= 128):
        xblock = 16
    else:
        xblock = 8
    blocks = (c.shape[0] + xblock - 1) // xblock
    _reduce_vec4_kernel(partials, c, fuse_relu).launch(
        grid=(blocks, 1, 1),
        block=(32, 1, 1),
    )


def _load_tensorop_gemm_module():
    path = Path(__file__).with_name("cutedsl_tensorop_gemm_v4.py")
    if not path.exists():
        path = Path("/tmp/cutedsl_tensorop_gemm_v4.py")
    if not path.exists():
        raise RuntimeError(f"missing patched CuteDSL tensorop GEMM helper: {path}")
    spec = importlib.util.spec_from_file_location("_cutedsl_tensorop_gemm_v4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CuteDSL tensorop GEMM helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_tensorop_gemm = _load_tensorop_gemm_module()


def partials_dtype(module_suite_name: str, input_dtype: torch.dtype) -> torch.dtype:
    del module_suite_name, input_dtype
    return torch.float32


def inductor_like_splits(m: int, n: int, k: int, limit: int) -> list[int]:
    del limit
    max_split = min(k // m, k // n)
    splits = [
        split
        for split in K_SPLITS
        if split <= max_split and k % split == 0 and k // split >= 16
    ]
    for split in _baseline_splits(m, n, k, 16):
        if split not in splits:
            splits.append(split)
    for k_per_split in (128, 256, 512):
        if k % k_per_split == 0:
            split = k // k_per_split
            if split not in splits:
                splits.append(split)
    return [split for split in splits if k // split in (128, 256, 512)]


def candidate_configs(split_values: list[int]) -> list[KernelConfig]:
    configs = []
    seen = set()
    tiles = [
        (16, 16, 32, 1, 5),
        (16, 16, 64, 1, 4),
        (16, 16, 64, 1, 3),
        (16, 16, 128, 1, 3),
        (16, 16, 256, 1, 3),
        (16, 32, 64, 1, 4),
        (32, 16, 64, 1, 4),
        (32, 32, 64, 1, 4),
        (32, 32, 64, 4, 3),
        (32, 32, 128, 4, 3),
        (32, 64, 64, 2, 3),
        (32, 64, 64, 4, 3),
    ]
    for split_k in split_values:
        for block_m, block_n, block_k, num_warps, num_stages in tiles:
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
                    8,
                    num_warps,
                    num_stages,
                )
            )
    return configs


def _atom_layout(num_warps: int) -> tuple[int, int, int]:
    if num_warps == 1:
        return (1, 1, 1)
    if num_warps == 2:
        return (1, 2, 1)
    if num_warps == 4:
        return (2, 2, 1)
    raise ValueError(f"unsupported CuteDSL atom layout for {num_warps} warps")


def _adapt_tensor(
    tensor: torch.Tensor,
    *,
    leading_dim: int,
    divisibility: int,
):
    key = (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.index,
        leading_dim,
        divisibility,
    )
    adapted = _TENSOR_CACHE.get(key)
    if adapted is None:
        del leading_dim, divisibility
        adapted = from_dlpack(tensor, assumed_align=16)
        _TENSOR_CACHE[key] = adapted
    return adapted


def _make_split_views(
    a: torch.Tensor,
    b: torch.Tensor,
    partials: torch.Tensor,
    split_k: int,
):
    m, k = a.shape
    n = b.shape[1]
    k_per_split = k // split_k
    a_view = torch.as_strided(a, (m, k_per_split, split_k), (k, 1, k_per_split))
    b_view = torch.as_strided(
        b, (n, k_per_split, split_k), (1, n, k_per_split * n)
    )
    partials_view = torch.as_strided(partials, (m, n, split_k), (n, 1, m * n))
    return (
        _adapt_tensor(a_view, leading_dim=1, divisibility=k_per_split),
        _adapt_tensor(b_view, leading_dim=0, divisibility=n),
        _adapt_tensor(partials_view, leading_dim=1, divisibility=n),
    )


def _compiled_gemm(
    config: KernelConfig,
    a_tensor,
    b_tensor,
    partials_tensor,
    input_dtype: torch.dtype,
):
    key = (
        config.split_k,
        config.block_m,
        config.block_n,
        config.block_k,
        config.num_warps,
        config.num_stages,
        input_dtype,
        int(a_tensor.shape[0]),
        int(a_tensor.shape[1]),
        int(b_tensor.shape[0]),
    )
    compiled = _GEMM_CACHE.get(key)
    if compiled is None:
        if input_dtype is torch.bfloat16:
            ab_dtype = cutlass.BFloat16
        elif input_dtype is torch.float16:
            ab_dtype = cutlass.Float16
        else:
            raise TypeError(f"unsupported CuteDSL tensorop input dtype: {input_dtype}")
        atom_layout = _atom_layout(config.num_warps)
        gemm = _tensorop_gemm.TensorOpGemm(
            ab_dtype,
            cutlass.Float32,
            cutlass.Float32,
            atom_layout,
        )
        gemm.cta_tiler = (config.block_m, config.block_n, config.block_k)
        gemm.num_stages = config.num_stages
        gemm.atom_layout_mnk = atom_layout
        gemm.num_threads = atom_layout[0] * atom_layout[1] * atom_layout[2] * 32
        gemm.bM, gemm.bN, gemm.bK = gemm.cta_tiler
        compiled = _tensorop_gemm.cute.compile(gemm, a_tensor, b_tensor, partials_tensor)
        _GEMM_CACHE[key] = compiled
    return compiled


def _compiled_vec4_reducer(
    split_k: int,
    output_dtype: torch.dtype,
    fuse_relu: bool,
    m: int,
    n: int,
):
    key = (split_k, output_dtype, fuse_relu, m, n, "vec4_tvm_ffi")
    reducer = _VEC4_REDUCER_CACHE.get(key)
    if reducer is None:
        if output_dtype is torch.bfloat16:
            c_dtype = cute.BFloat16
        elif output_dtype is torch.float16:
            c_dtype = cute.Float16
        elif output_dtype is torch.float32:
            c_dtype = cute.Float32
        else:
            raise TypeError(f"unsupported reducer output dtype: {output_dtype}")
        partials = cute.runtime.make_fake_tensor(
            cute.Float32,
            (split_k, m, n),
            (m * n, n, 1),
            assumed_align=16,
        )
        c = cute.runtime.make_fake_tensor(
            c_dtype,
            (m * n,),
            (1,),
            assumed_align=16,
        )
        reducer = cute.compile(
            _reduce_vec4_launch,
            partials,
            c,
            fuse_relu,
            options="--enable-tvm-ffi",
        )
        _VEC4_REDUCER_CACHE[key] = reducer
    return reducer


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
    m, n = a.shape[0], b.shape[1]
    if (
        config.block_m > m
        or config.block_n > n
        or m % config.block_m != 0
        or n % config.block_n != 0
    ):
        raise ValueError("CuteDSL GEMM tile must exactly tile the output shape")
    if min(m, n) < 64 and (config.block_m > 16 or config.block_n > 16):
        raise ValueError("CuteDSL larger spatial tiles are only used for larger outputs")
    if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype:
        raise TypeError("CuteDSL split-K GEMM candidate supports BF16 or FP16 inputs")
    if partials.dtype is not torch.float32:
        raise TypeError("CuteDSL split-K GEMM candidate requires FP32 partials")

    a_tensor, b_tensor, partials_tensor = _make_split_views(
        a, b, partials, config.split_k
    )
    gemm = _compiled_gemm(config, a_tensor, b_tensor, partials_tensor, a.dtype)
    gemm(a_tensor, b_tensor, partials_tensor)

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
