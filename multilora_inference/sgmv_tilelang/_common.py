"""Shared helpers for SGMV TileLang kernels."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import tilelang
import tilelang.language as T

_TILELANG_GEMM_CACHE: Dict[Tuple[str, str, int, int, int, int, int], Callable] = {}

_TORCH_TO_TILELANG_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def torch_dtype_to_tilelang(dtype: torch.dtype) -> str:
    """Map a torch dtype to the corresponding TileLang string."""
    if dtype not in _TORCH_TO_TILELANG_DTYPES:
        raise TypeError(f"Unsupported dtype for TileLang kernel: {dtype}")
    return _TORCH_TO_TILELANG_DTYPES[dtype]


@tilelang.jit(out_idx=[-1])
def _build_gemm_kernel(
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    M = T.symbolic("M")

    @T.prim_func
    def gemm(
        A: T.Tensor[(M, K), dtype],
        B: T.Tensor[(K, N), dtype],
        C: T.Tensor[(M, N), dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def get_gemm_kernel(
    *,
    K: int,
    N: int,
    dtype: str,
    accum_dtype: str,
    block_M: int = 64,
    block_N: int = 128,
    block_K: int = 32,
) -> Callable:
    """Return (and cache) a TileLang GEMM kernel for the given configuration."""
    key = (dtype, accum_dtype, N, K, block_M, block_N, block_K)
    kernel = _TILELANG_GEMM_CACHE.get(key)
    if kernel is None:
        kernel = _build_gemm_kernel(N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)
        _TILELANG_GEMM_CACHE[key] = kernel
    return kernel


def materialize_metadata(
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy small metadata tensors to CPU int64 for inexpensive host-side loops."""
    return (
        num_tokens_per_lora.to(device="cpu", dtype=torch.int64),
        lora_token_start_loc.to(device="cpu", dtype=torch.int64),
        lora_ids.to(device="cpu", dtype=torch.int64),
    )
