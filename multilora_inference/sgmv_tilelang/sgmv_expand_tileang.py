"""TileLang implementation of the SGMV expand kernel."""

from __future__ import annotations

import torch

from ._common import get_gemm_kernel, materialize_metadata, torch_dtype_to_tilelang


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


def sgmv_expand(
    Y: torch.Tensor,
    X: torch.Tensor,
    W: torch.Tensor,
    indices: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    num_lora_adapters: int,
    accumulate: bool,
) -> None:
    """Apply LoRA expand step via TileLang GEMM tiles.

    Optimizations:
    - Pre-sort once by LoRA id and compute per-LoRA contiguous slices into a sorted buffer.
    - Avoid per-group index_select/index_copy in the loop; unsort once at the end.
    - Use index_add_ when accumulating to reduce loads and improve bandwidth.
    - Small-M fallback to torch.matmul for tiny slices.
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

    # Work on a sorted view so each LoRA group forms a contiguous range
    sorted_token_indices_long = token_indices_sorted_by_lora_ids.to(dtype=torch.int64)
    X_sorted = torch.index_select(X, 0, sorted_token_indices_long)
    Y_sorted = torch.zeros_like(X_sorted.new_empty((X_sorted.shape[0], Y.shape[1])))

    dtype_str = torch_dtype_to_tilelang(X.dtype)
    accum_dtype = "float32"

    _, rank = X.shape
    feature_out = Y.shape[1]

    # Metadata on CPU for fast slot iteration
    num_tokens_cpu, lora_token_start_cpu, lora_ids_cpu = materialize_metadata(
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
    )

    weight_transposed = W.transpose(1, 2).contiguous()  # (L, rank, feature_out)

    kernel = get_gemm_kernel(
        K=rank,
        N=feature_out,
        dtype=dtype_str,
        accum_dtype=accum_dtype,
    )

    max_slots = lora_ids_cpu.numel()
    SMALL_M_THRESHOLD = 32
    for slot in range(max_slots):
        lora_id = int(lora_ids_cpu[slot])
        if lora_id == num_lora_adapters:
            continue
        token_count = int(num_tokens_cpu[slot])
        if token_count == 0:
            continue

        start = int(lora_token_start_cpu[slot])
        end = start + token_count

        x_subset = X_sorted[start:end]
        w_slice = weight_transposed[lora_id]  # (rank, feature_out)

        if token_count <= SMALL_M_THRESHOLD:
            y_chunk = x_subset @ w_slice
        else:
            y_chunk = kernel(x_subset, w_slice)

        # Write into sorted buffer
        Y_sorted[start:end].copy_(y_chunk)

    # Unsort once
    if accumulate:
        # index_add_ to accumulate the computed contributions
        Y.index_add_(0, sorted_token_indices_long, Y_sorted)
    else:
        # We need to zero only those rows that have the sentinel id; easier: overwrite all
        # active rows and zero others. Compute a mask of active rows and scatter.
        Y.zero_()
        Y.index_copy_(0, sorted_token_indices_long, Y_sorted)
