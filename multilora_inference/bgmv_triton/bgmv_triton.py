from typing import Optional

import torch
import triton
import triton.language as tl
import pdb

# Track which autotune keys have already been tuned per device to avoid extra warmups
# Keys mirror the Triton autotune keys for each kernel: SHRINK -> F_IN; EXPAND -> (F_IN, F_OUT)
_TUNED_SHRINK_KEYS = set()  # {(device_index, F_IN)}
_TUNED_EXPAND_KEYS = set()  # {(device_index, F_IN, F_OUT)}

def _cdiv(a, b):
    return (a + b - 1) // b


# Each program computes a single output Y[b, j]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 256}, num_warps=8, num_stages=3),
    ],
    key=['F_IN'],
    reset_to_zero=['Y_ptr']
)
@triton.jit
def bgmv_shrink_kernel(
    Y_ptr, X_ptr, W_ptr, indices_ptr,
    scale,                          # scalar float
    F_IN: tl.constexpr,             # int
    F_OUT: tl.constexpr,            # int
    NUM_LAYERS: tl.constexpr,
    LAYER_IDX: tl.constexpr,
    B,                              # int (batch size)
    SEQ_LEN,                        # int (sequence length)
    NUM_LORA_ADAPTERS: tl.constexpr, # int
    OUT_IS_FP16: tl.constexpr,
    OUT_IS_BF16: tl.constexpr,
    ADD_TO_Y: tl.constexpr,         # bool: if True -> Y += result ; else Y = result
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)  # batch b (map large dim to grid.x)
    pid_j = tl.program_id(axis=1)  # output row j (grid.y)

    b_seq  = pid_b // SEQ_LEN 

    j_in = pid_j < F_OUT
    b_in = pid_b < B

    # idx = indices[b] * num_layers + layer_idx
    idx_b = tl.load(indices_ptr + b_seq, mask=b_in, other=0)

    if idx_b == NUM_LORA_ADAPTERS:
        return
    
    idx = idx_b * NUM_LAYERS + LAYER_IDX

    # Pointer bases (use 64-bit offsets)
    idx64 = idx.to(tl.int64)
    j64 = pid_j.to(tl.int64)
    b64 = pid_b.to(tl.int64)
    FOUT64 = tl.full((), F_OUT, dtype=tl.int64)
    FIN64 = tl.full((), F_IN, dtype=tl.int64)

    w_row_base = (idx64 * FOUT64 + j64) * FIN64
    x_base = b64 * FIN64

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    k0 = tl.arange(0, BLOCK_K) # For vectorized operations
    for k_off in tl.range(0, F_IN, BLOCK_K, num_stages=2):
        k = k_off + k0
        k_mask = k < F_IN
        k64 = k.to(tl.int64)

        w = tl.load(W_ptr + w_row_base + k64, mask=j_in & k_mask, other=0).to(tl.float32)
        x = tl.load(X_ptr + x_base + k64, mask=b_in & k_mask, other=0).to(tl.float32)
        acc += tl.sum(w * x, axis=0)

    acc = acc * tl.full((), scale, dtype=tl.float32)

    # Writeback (use 64-bit offsets)
    y_ptr = Y_ptr + (pid_b.to(tl.int64) * tl.full((), F_OUT, dtype=tl.int64) + pid_j.to(tl.int64))
    
    if ADD_TO_Y:
        y_old = tl.load(y_ptr, mask=b_in & j_in, other=0).to(tl.float32)
        out = y_old + acc
    else:
        out = acc

    # Cast to output dtype explicitly
    if OUT_IS_FP16:
        out = out.to(tl.float16)
    elif OUT_IS_BF16:
        out = out.to(tl.bfloat16)
    else:
        out = out.to(tl.float32)

    tl.store(y_ptr, out, mask=b_in & j_in)


# Each program computes a tile of outputs Y[b, j0 : j0+BLOCK_M)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 8},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 16},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
    ],
    key=['F_IN', 'F_OUT'],
    restore_value=['Y_ptr']
)
@triton.jit
def bgmv_expand_kernel(
    Y_ptr, X_ptr, W_ptr, indices_ptr,
    scale,                          # scalar float
    F_IN: tl.constexpr,             # int
    F_OUT: tl.constexpr,            # int
    NUM_LAYERS: tl.constexpr,
    LAYER_IDX: tl.constexpr,
    B,                              # int (batch size)
    SEQ_LEN,                        # int (sequence length)
    NUM_LORA_ADAPTERS: tl.constexpr, # int
    OUT_IS_FP16: tl.constexpr,
    OUT_IS_BF16: tl.constexpr,
    ADD_TO_Y: tl.constexpr,         # bool: if True -> Y += result ; else Y = result
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)  # batch id (grid.x)
    pid_m = tl.program_id(axis=1)  # tile id along output rows (grid.y)

    b_seq  = pid_b // SEQ_LEN 

    j0 = pid_m * BLOCK_M
    offs_m = j0 + tl.arange(0, BLOCK_M)
    m_mask = offs_m < F_OUT

    # idx = indices[b] * num_layers + layer_idx
    idx_b = tl.load(indices_ptr + b_seq)

    if idx_b == NUM_LORA_ADAPTERS:
        return

    idx = idx_b * NUM_LAYERS + LAYER_IDX

    # 64-bit constants
    FIN64 = tl.full((), F_IN, dtype=tl.int64)
    FOUT64 = tl.full((), F_OUT, dtype=tl.int64)
    idx64 = idx.to(tl.int64)
    b64 = pid_b.to(tl.int64)

    # Base pointers
    x_base = b64 * FIN64
    w_base0 = (idx64 * FOUT64 + j0.to(tl.int64)) * FIN64  # base for row j0

    # Accumulator [BLOCK_M]
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    k0 = tl.arange(0, BLOCK_K)
    for k_off in tl.range(0, F_IN, BLOCK_K, num_stages=2):
        k = k_off + k0
        k_mask = k < F_IN
        k64 = k.to(tl.int64)

        # X: [BLOCK_K]
        x = tl.load(X_ptr + x_base + k64, mask=k_mask, other=0).to(tl.float32)

        # W block: [BLOCK_M, BLOCK_K]
        row_offsets = (tl.arange(0, BLOCK_M).to(tl.int64) * FIN64)[:, None]
        w_ptrs = W_ptr + (w_base0 + row_offsets) + k64[None, :]
        w = tl.load(w_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0).to(tl.float32)

        # Reduction across K
        acc += tl.sum(w * x[None, :], axis=1)

    acc = acc * tl.full((), scale, dtype=tl.float32)

    # Write back to Y[b, offs_m] (use 64-bit offsets)
    y_ptr = Y_ptr + (pid_b.to(tl.int64) * FOUT64 + offs_m.to(tl.int64))
    
    if ADD_TO_Y:
        y_old = tl.load(y_ptr, mask=m_mask, other=0).to(tl.float32)
        out = y_old + acc
    else:
        out = acc

    if OUT_IS_FP16:
        out = out.to(tl.float16)
    elif OUT_IS_BF16:
        out = out.to(tl.bfloat16)
    else:
        out = out.to(tl.float32)

    tl.store(y_ptr, out, mask=m_mask)


def bgmv_triton(
    Y: torch.Tensor,      # [B, F_out], dtype in {float32, float16, bfloat16}, contiguous
    X: torch.Tensor,      # [B, F_in],  same dtype as Y, contiguous
    W: torch.Tensor,      # [L * num_layers, F_out, F_in], contiguous
    indices: torch.Tensor,# [B], int64/int64; values in [0, L)
    *,
    seq_len: int,
    num_layers: int,
    layer_idx: int,
    num_lora_adapters: int,
    scale: float = 1.0,
    accumulate: bool = False,   # if True: Y += result; else Y = result
):
    """
    Computes: out[b, j] = sum_i W[idx(b), j, i] * X[b, i] * scale
    where idx(b) = indices[b] * num_layers + layer_idx

    If accumulate=True, the result is added into Y; otherwise Y is overwritten.
    """
    assert Y.is_cuda and X.is_cuda and W.is_cuda and indices.is_cuda
    assert Y.dtype == X.dtype == W.dtype
    assert Y.is_contiguous() and X.is_contiguous() and W.is_contiguous()

    SEQ_LEN = seq_len
    # If we receive 3-D tensors, flatten tokens
    if X.ndim == 3 and Y.ndim == 3:
        B, n, fin = X.shape
        _, n2, fout = Y.shape
        assert n == n2, "X and Y must have same sequence length"
        X = X.contiguous().view(B * n, fin)
        Y = Y.contiguous().view(B * n, fout)

        # SEQ_LEN = n

        ## No need to broadcast indices because that is handled in the kernel

        # Normalize indices length: accept [B] or [B*n]
        # if indices.ndim != 1:
        #     raise AssertionError("indices must be 1-D")
        # if indices.numel() == B:
        #     indices = indices.contiguous().repeat_interleave(n)
        # elif indices.numel() == B * n:
        #     indices = indices.contiguous()
        # else:
        #     raise AssertionError("indices must be length B (per sequence) or B*n (per token)")

    T, F_out = Y.shape
    Tx, F_in = X.shape
    assert T == Tx
    assert W.shape[1] == F_out and W.shape[2] == F_in
    assert 0 <= layer_idx < num_layers

    # dtype flags for downcast on store
    if Y.dtype == torch.float16:
        out_is_fp16, out_is_bf16 = True, False
    elif Y.dtype == torch.bfloat16:
        out_is_fp16, out_is_bf16 = False, True
    elif Y.dtype == torch.float32:
        out_is_fp16, out_is_bf16 = False, False
    else:
        raise TypeError(f"Unsupported dtype {Y.dtype}")

    F_IN_i = int(F_in)
    F_OUT_i = int(F_out)
    B_i = int(T)
    SEQ_LEN_i = int(SEQ_LEN)

    if F_in < F_out:
        # EXPAND
        def grid(meta):
            # map large flattened batch (B_i) to grid.x to avoid CUDA grid.y limits
            return (B_i, _cdiv(F_OUT_i, meta['BLOCK_M']))
                # Warmup only once per autotune key (device, F_IN, F_OUT)
        # if accumulate:
        #     device_idx = Y.device.index if Y.is_cuda else -1
        #     key = (device_idx, F_IN_i, F_OUT_i)
        #     if key not in _TUNED_EXPAND_KEYS:
        #         tmpY = torch.empty_like(Y)
        #         bgmv_expand_kernel[grid](
        #             tmpY, X, W, indices,
        #             float(scale),
        #             F_IN_i, F_OUT_i, int(num_layers), int(layer_idx),
        #             B_i,
        #             SEQ_LEN_i,
        #             int(num_lora_adapters),
        #             out_is_fp16, out_is_bf16,
        #             False,
        #         )
        #         _TUNED_EXPAND_KEYS.add(key)
        bgmv_expand_kernel[grid](
            Y, X, W, indices,
            float(scale),
            F_IN_i, F_OUT_i, int(num_layers), int(layer_idx),
            B_i,
            SEQ_LEN_i,
            int(num_lora_adapters),
            out_is_fp16, out_is_bf16,
            bool(accumulate),
        )
    else:
        # SHRINK
        def grid(meta):
            # map large flattened batch (B_i) to grid.x to avoid CUDA grid.y limits
            return (B_i, F_OUT_i)
                # Warmup only once per autotune key (device, F_IN)
        # if accumulate:
        #     device_idx = Y.device.index if Y.is_cuda else -1
        #     key = (device_idx, F_IN_i)
        #     if key not in _TUNED_SHRINK_KEYS:
        #         tmpY = torch.empty_like(Y)
        #         bgmv_shrink_kernel[grid](
        #             tmpY, X, W, indices,
        #             float(scale),
        #             F_IN_i, F_OUT_i, int(num_layers), int(layer_idx),
        #             B_i,
        #             SEQ_LEN_i,
        #             int(num_lora_adapters),
        #             out_is_fp16, out_is_bf16,
        #             False,
        #         )
        #         _TUNED_SHRINK_KEYS.add(key)
        bgmv_shrink_kernel[grid](
            Y, X, W, indices,
            float(scale),
            F_IN_i, F_OUT_i, int(num_layers), int(layer_idx),
            B_i,
            SEQ_LEN_i,
            int(num_lora_adapters),
            out_is_fp16, out_is_bf16,
            bool(accumulate),
        )