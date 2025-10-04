import torch
import triton
import triton.language as tl

from .sgmv_shrink_triton import sgmv_shrink
from .sgmv_expand_triton import sgmv_expand

def lora_sgmv_triton(
    Y: torch.Tensor,      # [B, F_out], dtype in {float32, float16, bfloat16}, contiguous
    X: torch.Tensor,      # [B, F_in],  same dtype as Y, contiguous
    A: torch.Tensor,      # [L, r, F_in], contiguous
    B: torch.Tensor,      # [L, F_out, r], contiguous
    indices: torch.Tensor,# [B], int64/int64; values in [0, L)
    num_lora_adapters: int,
    *,
    scale: float = 1.0,
    accumulate: bool = True,   # if True: Y += result; else Y = result
):
    batch_size = X.shape[0]
    if indices.numel() == batch_size:
        indices = indices.repeat_interleave(X.shape[1])
    indices = indices.contiguous()

    X_shape = X.shape
    Y_shape = Y.shape
    has_sequence_dim = X.dim() == 3
    decode_mode = has_sequence_dim and X_shape[-2] == 1

    # Flatten inputs (LoRA weights already contiguous from loading)
    X = X.view(-1, X_shape[-1])
    Y = Y.view(-1, Y_shape[-1])

    # Calculate the token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc
    num_tokens_per_lora = torch.zeros(num_lora_adapters + 1, dtype=indices.dtype, device=indices.device)
    lora_token_start_loc = torch.zeros(num_lora_adapters + 2, dtype=indices.dtype, device=indices.device)
    active_lora_ids = torch.ones(num_lora_adapters + 1, dtype=indices.dtype, device=indices.device) * num_lora_adapters

    token_indices_sorted_by_lora_ids = indices.argsort(stable=True)
    lora_ids, num_tokens_per_lora_curr = torch.unique(indices, sorted=True, return_counts=True)
    active_lora_ids[:lora_ids.shape[0]] = lora_ids
    num_tokens_per_lora[:num_tokens_per_lora_curr.shape[0]] = num_tokens_per_lora_curr
    lora_token_start_loc[1: 1+num_tokens_per_lora_curr.shape[0]] = torch.cumsum(num_tokens_per_lora_curr, dim=0)

    Y_intermediate = torch.zeros(X.shape[0], A.shape[1], dtype=Y.dtype, device=Y.device)

    sgmv_shrink(
        Y_intermediate,
        X,
        A,
        indices,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        num_lora_adapters,
        scale,
        decode_mode=decode_mode,
    )
    sgmv_expand(
        Y,
        Y_intermediate,
        B,
        indices,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        num_lora_adapters,
        accumulate,
        decode_mode=decode_mode,
    )
