"""TileLang implementation of the SGMV shrink kernel."""

from __future__ import annotations

import torch

from ._common import get_gemm_kernel, materialize_metadata, torch_dtype_to_tilelang


def _as_float(value: float | torch.Tensor) -> float:
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


def _validate_inputs(
    Y: torch.Tensor,
    X: torch.Tensor,
    W: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
) -> None:
    if not (Y.is_cuda and X.is_cuda and W.is_cuda):
        raise AssertionError("All inputs must be CUDA tensors")
    if not (token_indices_sorted_by_lora_ids.is_cuda and num_tokens_per_lora.is_cuda and lora_token_start_loc.is_cuda and lora_ids.is_cuda):
        raise AssertionError("Metadata tensors must be CUDA tensors")
    if not (Y.is_contiguous() and X.is_contiguous() and W.is_contiguous()):
        raise AssertionError("Y, X, and W must be contiguous")
    if Y.shape[0] != X.shape[0]:
        raise AssertionError("Batch dimension mismatch between Y and X")
    if token_indices_sorted_by_lora_ids.numel() != X.shape[0]:
        raise AssertionError("token_indices_sorted_by_lora_ids length mismatch")
    if Y.shape[1] != W.shape[1] or X.shape[1] != W.shape[2]:
        raise AssertionError("Weight dimensions do not align with inputs")
    if num_tokens_per_lora.numel() != lora_ids.numel():
        raise AssertionError("num_tokens_per_lora and lora_ids must match in length")
    if lora_token_start_loc.numel() != lora_ids.numel() + 1:
        raise AssertionError("lora_token_start_loc must have one extra element for the prefix sum")


def sgmv_shrink(
    Y: torch.Tensor,
    X: torch.Tensor,
    W: torch.Tensor,
    indices: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    num_lora_adapters: int,
    scale: float | torch.Tensor,
) -> None:
    """Populate ``Y`` with LoRA shrink results computed via TileLang GEMM tiles.

    Optimizations:
    - Single pre-sort of tokens by LoRA id to make per-LoRA slices contiguous.
    - Avoid per-group index_select/index_copy inside the loop; compute into a sorted buffer
      and unsort once at the end.
    - Small-M fallback to torch.matmul to reduce kernel-launch overhead on tiny groups.
    """
    _validate_inputs(
        Y,
        X,
        W,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
    )

    if X.shape[0] != indices.numel():
        raise AssertionError("indices must cover every token")

    dtype_str = torch_dtype_to_tilelang(X.dtype)
    accum_dtype = "float32"
    scale_value = _as_float(scale)

    batch_tokens, feature_in = X.shape
    rank = Y.shape[1]

    # Work on a sorted view so each LoRA group is a contiguous range
    sorted_token_indices_long = token_indices_sorted_by_lora_ids.to(dtype=torch.int64)
    X_sorted = torch.index_select(X, 0, sorted_token_indices_long)
    Y_sorted = torch.empty_like(Y)
    num_tokens_cpu, lora_token_start_cpu, lora_ids_cpu = materialize_metadata(
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
    )

    weight_transposed = W.transpose(1, 2).contiguous()  # (L, feature_in, rank)

    kernel = get_gemm_kernel(
        K=feature_in,
        N=rank,
        dtype=dtype_str,
        accum_dtype=accum_dtype,
    )

    # Kernel for typical sizes
    max_slots = lora_ids_cpu.numel()
    SMALL_M_THRESHOLD = 32  # use eager matmul for very small M to cut overhead
    for slot in range(max_slots):
        lora_id = int(lora_ids_cpu[slot])
        if lora_id == num_lora_adapters:
            continue
        token_count = int(num_tokens_cpu[slot])
        if token_count == 0:
            continue

        start = int(lora_token_start_cpu[slot])
        end = start + token_count

        # Contiguous slice in the sorted tensors
        x_subset = X_sorted[start:end]
        w_slice = weight_transposed[lora_id]  # (feature_in, rank)

        # Choose the most efficient path for the slice size
        if token_count <= SMALL_M_THRESHOLD:
            y_chunk = x_subset @ w_slice
        else:
            y_chunk = kernel(x_subset, w_slice)
        if scale_value != 1.0:
            y_chunk.mul_(scale_value)

        # Write back to the sorted buffer directly
        Y_sorted[start:end].copy_(y_chunk)

    # Unsort once at the end
    Y.index_copy_(0, sorted_token_indices_long, Y_sorted)
