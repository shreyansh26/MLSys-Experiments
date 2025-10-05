import torch

from .sgmv_shrink_tilelang import sgmv_shrink
from .sgmv_expand_tileang import sgmv_expand


def lora_sgmv_tilelang(
    Y: torch.Tensor,      # [B, F_out], dtype in {float32, float16, bfloat16}, contiguous
    X: torch.Tensor,      # [B, F_in] or [B, 1, F_in], same dtype as Y, contiguous
    A: torch.Tensor,      # [L, r, F_in], contiguous
    B: torch.Tensor,      # [L, F_out, r], contiguous
    indices: torch.Tensor,# [B] or [B * tokens_per_batch], int32/int64; values in [0, L] with sentinel L
    num_lora_adapters: int,
    *,
    scale: float = 1.0,
    accumulate: bool = True,   # if True: Y += result; else Y = result
) -> None:
    """
    TileLang SGMV pipeline: shrink (X @ A^T) then expand (.. @ B^T) with grouping by LoRA id.

    - indices may contain a sentinel value equal to num_lora_adapters, which marks rows with no LoRA; these rows contribute zeros in both phases.
    - When accumulate is False, expand will zero the output rows not written (i.e., sentinel rows) and overwrite active rows.
    """
    batch_size = X.shape[0]
    if indices.numel() == batch_size:
        # If X has an extra tokens-per-batch dimension (e.g., [B, 1, F_in]),
        # repeat indices accordingly so it matches the first dim after the view below.
        indices = indices.repeat_interleave(X.shape[1] if X.ndim == 3 else 1)
    indices = indices.contiguous()

    # Flatten last two dims if present (e.g., [B, 1, F_in] -> [B, F_in])
    X_shape = X.shape
    Y_shape = Y.shape
    X = X.view(-1, X_shape[-1])
    Y = Y.view(-1, Y_shape[-1])

    # Build grouping metadata
    num_tokens_per_lora = torch.zeros(num_lora_adapters + 1, dtype=indices.dtype, device=indices.device)
    lora_token_start_loc = torch.zeros(num_lora_adapters + 2, dtype=indices.dtype, device=indices.device)
    active_lora_ids = torch.ones(num_lora_adapters + 1, dtype=indices.dtype, device=indices.device) * num_lora_adapters

    token_indices_sorted_by_lora_ids = indices.argsort(stable=True)
    lora_ids, num_tokens_per_lora_curr = torch.unique(indices, sorted=True, return_counts=True)
    active_lora_ids[: lora_ids.shape[0]] = lora_ids
    num_tokens_per_lora[: num_tokens_per_lora_curr.shape[0]] = num_tokens_per_lora_curr
    lora_token_start_loc[1 : 1 + num_tokens_per_lora_curr.shape[0]] = torch.cumsum(num_tokens_per_lora_curr, dim=0)

    # Intermediate buffer for shrink output (rank dimension)
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
    )



