import torch
import triton
import triton.language as tl


# Local cache for LoRA-B pointer/stride info when using a single tensor.
_LORA_B_PTR_CACHE: dict[int, tuple[torch.Tensor, int, int, int]] = {}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['K', 'N'],
    restore_value=['Y_ptr']
)
@triton.jit
def sgmv_expand_decode_kernel(
    Y_ptr, X_ptr, lora_ptr_tensor,
    M, N, K,
    indices,
    NUM_LORA_ADAPTERS,
    X_ptr_stride_d0, X_ptr_stride_d1,
    lora_stride_d0, lora_stride_d1, lora_stride_d2,
    Y_ptr_stride_d0, Y_ptr_stride_d1,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_n = tl.program_id(1)

    token_id = pid_token
    if token_id >= M:
        return

    lora_id = tl.load(indices + token_id)
    if lora_id == NUM_LORA_ADAPTERS:
        return

    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offset_n < N

    k_offsets = tl.arange(0, BLOCK_K)

    x_base_ptr = X_ptr + token_id * X_ptr_stride_d0
    y_base_ptr = Y_ptr + token_id * Y_ptr_stride_d0
    lora_base_ptr = lora_ptr_tensor + lora_stride_d0 * lora_id

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_indices = k_start + k_offsets
        k_mask = k_indices < K

        x_vals = tl.load(
            x_base_ptr + k_indices * X_ptr_stride_d1,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        lora_ptr = (
            lora_base_ptr
            + offset_n[None, :] * lora_stride_d1
            + k_indices[:, None] * lora_stride_d2
        )

        lora_vals = tl.load(
            lora_ptr,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(lora_vals * x_vals[:, None], axis=0)

    if ACCUMULATE:
        y_prev = tl.load(
            y_base_ptr + offset_n * Y_ptr_stride_d1,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        acc += y_prev

    acc_out = acc.to(Y_ptr.dtype.element_ty)
    tl.store(
        y_base_ptr + offset_n * Y_ptr_stride_d1,
        acc_out,
        mask=n_mask,
    )


def _get_lora_b_ptr(W_ptr: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    """
    Simplified variant of vLLM's _get_lora_b_ptr for a single LoRA-A tensor.

    Accepts a single tensor with shape (lora_num, size, rank) or
    (lora_num, 1, size, rank). If the second dimension is 1, it is squeezed.

    Returns a tuple of:
      - normalized contiguous tensor with shape (lora_num, size, rank)
      - stride along dim 0
      - stride along dim 1
      - stride along dim 2

    The result is cached by the tensor's data_ptr() for reuse.
    """
    assert isinstance(W_ptr, torch.Tensor), "W_ptr must be a torch.Tensor"

    lora_b = W_ptr
    if lora_b.ndim == 4:  # shape: (lora_num, 1, size, rank)
        assert lora_b.size(1) == 1, "Expected LoRA-A shape (lora_num, 1, size, rank) with dim1 == 1"
        lora_b = lora_b.squeeze(dim=1)
    else:
        assert lora_b.ndim == 3, "Expected LoRA-A shape (lora_num, size, rank)"

    assert lora_b.is_contiguous(), "LoRA-A tensor must be contiguous"

    key = lora_b.data_ptr()
    if key in _LORA_B_PTR_CACHE:
        return _LORA_B_PTR_CACHE[key]

    stride_d0 = lora_b.stride(0)
    stride_d1 = lora_b.stride(1)
    stride_d2 = lora_b.stride(2)

    result = (lora_b, stride_d0, stride_d1, stride_d2)
    _LORA_B_PTR_CACHE[key] = result
    return result


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['K', 'N'],
    restore_value=['Y_ptr']
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def sgmv_expand_kernel(
    Y_ptr, X_ptr, lora_ptr_tensor,
    M, N, K,
    indices,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    NUM_LORA_ADAPTERS,
    X_ptr_stride_d0, X_ptr_stride_d1,
    lora_stride_d0, lora_stride_d1, lora_stride_d2,
    Y_ptr_stride_d0, Y_ptr_stride_d1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    blocks_n = tl.cdiv(N, BLOCK_N)
    blocks_m = tl.cdiv(M, BLOCK_M)
    
    pid_m_n = tl.program_id(0)
    pid_lora_idx = tl.program_id(1)

    pid_m = pid_m_n % blocks_m
    pid_n = (pid_m_n // blocks_m) % blocks_n

    lora_id = tl.load(lora_ids + pid_lora_idx)

    if lora_id == NUM_LORA_ADAPTERS:
        return

    lora_m_size = tl.load(num_tokens_per_lora + pid_lora_idx)

    block_m_offset = pid_m * BLOCK_M
    # If offset is past the num tokens for that lora
    if block_m_offset >= lora_m_size:
        return

    if pid_n * BLOCK_N >= N:
        return
    
    num_rows_to_process = min(BLOCK_M, lora_m_size - block_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + pid_lora_idx)
    # get token indices for that lora idx
    block_lora_seq_indices = token_indices_sorted_by_lora_ids + lora_m_indices_start + block_m_offset

    # load token indices for that lora idx
    offset_m = tl.arange(0, BLOCK_M)
    row_mask = offset_m < num_rows_to_process
    row_address_map = tl.load(block_lora_seq_indices + offset_m, mask=row_mask, other=0)

    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_n = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    offset_k = tl.arange(0, BLOCK_K)

    x_ptr = X_ptr + row_address_map[:, None] * X_ptr_stride_d0 + offset_k[None, :] * X_ptr_stride_d1
    lora_ptr = lora_ptr_tensor + lora_id * lora_stride_d0 + offset_n[None, :] * lora_stride_d1 + offset_k[:, None] * lora_stride_d2

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            tiled_x = tl.load(x_ptr, mask=row_mask[:, None], other=0)
            tiled_lora = tl.load(lora_ptr)
        else:
            remaining_k = K - k * BLOCK_K
            k_mask = offset_k[None, :] < remaining_k
            tiled_x = tl.load(x_ptr, mask=row_mask[:, None] & k_mask, other=0)
            tiled_lora = tl.load(lora_ptr, mask=offset_k[:, None] < remaining_k, other=0)
        
        if X_ptr.dtype.element_ty != lora_ptr_tensor.dtype.element_ty:
            tiled_x = tiled_x.to(lora_ptr_tensor.dtype.element_ty)
        
        acc += tl.dot(tiled_x, tiled_lora)

        x_ptr += BLOCK_K * X_ptr_stride_d1
        lora_ptr += BLOCK_K * lora_stride_d2

    offset_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_ym = tl.arange(0, BLOCK_M)
    offset_yn = tl.max_contiguous(tl.multiple_of(offset_yn % N, BLOCK_N), BLOCK_N)

    y_ptr = Y_ptr + row_address_map[:, None] * Y_ptr_stride_d0 + offset_yn[None, :] * Y_ptr_stride_d1
    y_mask = (offset_ym[:, None] < num_rows_to_process) & (offset_yn[None, :] < N)
  
    acc_out = acc.to(lora_ptr_tensor.dtype.element_ty)
    if ACCUMULATE:
        tiled_out = tl.load(y_ptr, mask=y_mask, other=0)
        acc_out = acc_out + tiled_out
        
    tl.store(y_ptr, acc_out, mask=y_mask)


def sgmv_expand(
    Y_ptr, X_ptr, W_ptr,
    indices,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    num_lora_adapters,
    accumulate,
    decode_mode: bool = False,
):
    assert X_ptr.is_cuda and W_ptr.is_cuda and token_indices_sorted_by_lora_ids.is_cuda and num_tokens_per_lora.is_cuda and lora_token_start_loc.is_cuda
    assert X_ptr.is_contiguous() and W_ptr.is_contiguous() and Y_ptr.is_contiguous()

    # F_OUT is rank here
    T, F_in = X_ptr.shape
    _, F_out = Y_ptr.shape
    NL, W_F_out, W_F_in = W_ptr.shape
    assert F_out == W_F_out and F_in == W_F_in

    assert T == indices.numel()
    assert T == token_indices_sorted_by_lora_ids.numel()

    assert num_tokens_per_lora.numel() == lora_ids.numel()
    assert lora_token_start_loc.numel() == lora_ids.numel() + 1

    lora_ptr_tensor, lora_stride_d0, lora_stride_d1, lora_stride_d2 = _get_lora_b_ptr(W_ptr)
    MAX_LORAS = lora_ids.numel()

    M = T
    N = W_F_out
    K = W_F_in

    if decode_mode:
        grid_decode = lambda meta: (
            M,
            triton.cdiv(N, meta['BLOCK_N']),
        )

        sgmv_expand_decode_kernel[grid_decode](
            Y_ptr, X_ptr, lora_ptr_tensor,
            M, N, K,
            indices,
            num_lora_adapters,
            X_ptr.stride(0), X_ptr.stride(1),
            lora_stride_d0, lora_stride_d1, lora_stride_d2,
            Y_ptr.stride(0), Y_ptr.stride(1),
            ACCUMULATE=bool(accumulate),
        )
        return

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        MAX_LORAS,
    )

    sgmv_expand_kernel[grid](
        Y_ptr, X_ptr, lora_ptr_tensor,
        M, N, K,
        indices,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        num_lora_adapters,
        X_ptr.stride(0), X_ptr.stride(1),
        lora_stride_d0, lora_stride_d1, lora_stride_d2,
        Y_ptr.stride(0), Y_ptr.stride(1),
        ACCUMULATE=bool(accumulate)
    )
