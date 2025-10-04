import torch
import triton
import triton.language as tl
from triton.language.core import validate_block_shape


# Local cache for LoRA-A pointer/stride info when using a single tensor.
_LORA_A_PTR_CACHE: dict[int, tuple[torch.Tensor, int, int, int]] = {}


def _get_lora_a_ptr(W_ptr: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    """
    Simplified variant of vLLM's _get_lora_a_ptr for a single LoRA-A tensor.

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

    lora_a = W_ptr
    if lora_a.ndim == 4:  # shape: (lora_num, 1, size, rank)
        assert lora_a.size(1) == 1, "Expected LoRA-A shape (lora_num, 1, size, rank) with dim1 == 1"
        lora_a = lora_a.squeeze(dim=1)
    else:
        assert lora_a.ndim == 3, "Expected LoRA-A shape (lora_num, size, rank)"

    assert lora_a.is_contiguous(), "LoRA-A tensor must be contiguous"

    key = lora_a.data_ptr()
    if key in _LORA_A_PTR_CACHE:
        return _LORA_A_PTR_CACHE[key]

    stride_d0 = lora_a.stride(0)
    stride_d1 = lora_a.stride(1)
    stride_d2 = lora_a.stride(2)

    result = (lora_a, stride_d0, stride_d1, stride_d2)
    _LORA_A_PTR_CACHE[key] = result
    return result


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 256, 'SPLIT_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 128, 'SPLIT_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 512, 'SPLIT_K': 8}, num_warps=4, num_stages=2),
    ],
    key=['K', 'N'],
    reset_to_zero=['Y_ptr']
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def sgmv_shrink_kernel(
    Y_ptr, X_ptr, lora_ptr_tensor,
    M, N, K,
    indices,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scale,
    NUM_LORA_ADAPTERS,
    X_ptr_stride_d0, X_ptr_stride_d1,
    lora_stride_d0, lora_stride_d1, lora_stride_d2,
    Y_ptr_stride_d0, Y_ptr_stride_d1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    blocks_n = tl.cdiv(N, BLOCK_N)
    blocks_m = tl.cdiv(M, BLOCK_M)
    
    pid_sk_m_n = tl.program_id(0)
    pid_lora_idx = tl.program_id(1)

    pid_sk = pid_sk_m_n % SPLIT_K
    pid_m = (pid_sk_m_n // SPLIT_K) % blocks_m
    pid_n = (pid_sk_m_n // (SPLIT_K * blocks_m)) % blocks_n

    lora_id = tl.load(lora_ids + pid_lora_idx)
    
    if lora_id == NUM_LORA_ADAPTERS:
        return

    lora_m_size = tl.load(num_tokens_per_lora + pid_lora_idx)

    block_m_offset = pid_m * BLOCK_M
    # If offset is past the num tokens for that lora
    if block_m_offset >= lora_m_size:
        return
    
    num_rows_to_process = min(BLOCK_M, lora_m_size - block_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + pid_lora_idx)
    # get token indices for that lora idx
    block_lora_seq_indices = token_indices_sorted_by_lora_ids + lora_m_indices_start + block_m_offset
    
    # load token indices for that lora idx
    offset_m = tl.arange(0, BLOCK_M) % num_rows_to_process
    # Repeated values, yes but are masked out in output y_mask
    row_address_map = tl.load(block_lora_seq_indices + offset_m)

    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_n = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)

    x_ptr = X_ptr + row_address_map[:, None] * X_ptr_stride_d0 + offset_k[None, :] * X_ptr_stride_d1
    lora_ptr = lora_ptr_tensor + lora_stride_d0 * lora_id + offset_n[None, :] * lora_stride_d1 + offset_k[:, None] * lora_stride_d2

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            tiled_x = tl.load(x_ptr)
            tiled_lora = tl.load(lora_ptr)
        else:
            tiled_x = tl.load(x_ptr, mask=offset_k[None, :] < K - k * (BLOCK_K * SPLIT_K), other=0)
            tiled_lora = tl.load(lora_ptr, mask=offset_k[:, None] < K - k * (BLOCK_K * SPLIT_K), other=0)
        
        acc += tl.dot(tiled_x, tiled_lora)

        x_ptr += BLOCK_K * SPLIT_K * X_ptr_stride_d1
        lora_ptr += BLOCK_K * SPLIT_K * lora_stride_d2

    offset_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_ym = tl.arange(0, BLOCK_M)
    offset_yn = tl.max_contiguous(tl.multiple_of(offset_yn % N, BLOCK_N), BLOCK_N)

    y_ptr = Y_ptr + row_address_map[:, None] * Y_ptr_stride_d0 + offset_yn[None, :] * Y_ptr_stride_d1
    y_mask = (offset_ym[:, None] < num_rows_to_process) & (offset_yn[None, :] < N)

    acc = acc * scale
    
    if SPLIT_K == 1:
        tl.store(y_ptr, acc, mask=y_mask)
    else:
        tl.atomic_add(y_ptr, acc, mask=y_mask)


def sgmv_shrink(
    Y_ptr, X_ptr, W_ptr, 
    indices,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    num_lora_adapters,
    scale,
):
    assert X_ptr.is_cuda and W_ptr.is_cuda and token_indices_sorted_by_lora_ids.is_cuda and num_tokens_per_lora.is_cuda and lora_token_start_loc.is_cuda
    assert X_ptr.dtype == W_ptr.dtype
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

    lora_ptr_tensor, lora_stride_d0, lora_stride_d1, lora_stride_d2 = _get_lora_a_ptr(W_ptr)
    MAX_LORAS = lora_ids.numel()

    M = T
    N = W_F_out
    K = W_F_in

    grid = lambda meta: (
        meta['SPLIT_K'] * triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        MAX_LORAS,
    )

    sgmv_shrink_kernel[grid](
        Y_ptr, X_ptr, lora_ptr_tensor,
        M, N, K,
        indices,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scale,
        num_lora_adapters,
        X_ptr.stride(0), X_ptr.stride(1),
        lora_stride_d0, lora_stride_d1, lora_stride_d2,
        Y_ptr.stride(0), Y_ptr.stride(1)
    )
    