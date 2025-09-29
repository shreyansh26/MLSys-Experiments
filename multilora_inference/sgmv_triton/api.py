import torch
import triton
import triton.language as tl

from .sgmv_triton import sgmv_shrink_kernel, sgmv_expand_kernel

def lora_sgmv_triton(
    Y: torch.Tensor,      # [B, F_out], dtype in {float32, float16, bfloat16}, contiguous
    X: torch.Tensor,      # [B, F_in],  same dtype as Y, contiguous
    A: torch.Tensor,      # [L, r, F_in], contiguous
    B: torch.Tensor,      # [L, F_out, r], contiguous
    indices: torch.Tensor,# [B], int64/int64; values in [0, L)
    *,
    num_lora_adapters: int,
    scale: float = 1.0,
    accumulate: bool = False,   # if True: Y += result; else Y = result
):
    # Calculate the indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc
    indices_sorted_by_lora_ids = indices.argsort()
    num_tokens_per_lora = torch.bincount(indices_sorted_by_lora_ids)
    lora_token_start_loc = torch.cumsum(num_tokens_per_lora, dim=0) - num_tokens_per_lora       # Exclusive prefix sum


    Y_intermediate = torch.zeros(X.shape[0], A.shape[1], dtype=Y.dtype, device=Y.device)

    sgmv_shrink_kernel(Y_intermediate, X, A, indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, scale)
    sgmv_expand_kernel(Y, Y_intermediate, B, indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, 1.0, accumulate)