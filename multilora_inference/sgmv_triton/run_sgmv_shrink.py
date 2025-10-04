import torch
from sgmv_shrink_triton import sgmv_shrink

B = 16
num_layers = 8
L = 10
layer_idx = 2
F_in = 16384
F_out = 16
scale = 0.25
n = 32

dtype = torch.float16  # or torch.bfloat16, torch.float32
device = 'cuda'

def prepare_metadata_and_call(Y, X, W, indices, num_layers, layer_idx, L, scale):
    """Helper to prepare metadata and call sgmv_shrink"""
    num_tokens_per_lora = torch.zeros(L + 1, dtype=indices.dtype, device=indices.device)
    lora_token_start_loc = torch.zeros(L + 2, dtype=indices.dtype, device=indices.device)
    active_lora_ids = torch.ones(L + 1, dtype=indices.dtype, device=indices.device) * L

    token_indices_sorted_by_lora_ids = indices.argsort(stable=True)
    lora_ids, num_tokens_per_lora_curr = torch.unique(indices, sorted=True, return_counts=True)
    active_lora_ids[:lora_ids.shape[0]] = lora_ids
    num_tokens_per_lora[:num_tokens_per_lora_curr.shape[0]] = num_tokens_per_lora_curr
    lora_token_start_loc[1: 1+num_tokens_per_lora_curr.shape[0]] = torch.cumsum(num_tokens_per_lora_curr, dim=0)

    w_sgmv = W[torch.arange(L, device=device) * num_layers + layer_idx]

    Y.zero_()
    sgmv_shrink(Y, X, w_sgmv, indices, token_indices_sorted_by_lora_ids, num_tokens_per_lora, 
                lora_token_start_loc, active_lora_ids, L, scale)

## 2D case
print("2D case")
X = torch.randn(B, F_in, device=device, dtype=dtype)
W = torch.randn(L * num_layers, F_out, F_in, device=device, dtype=dtype)
Y = torch.randn(B, F_out, device=device, dtype=dtype)
indices = torch.randint(low=0, high=L, size=(B,), device=device, dtype=torch.int32)

prepare_metadata_and_call(Y, X, W, indices, num_layers, layer_idx, L, scale)

# Correctness check vs. PyTorch
idx = indices.to(torch.long) * num_layers + layer_idx
W_sel = W[idx, :, :]
ref = torch.einsum("bfi,bi->bf", W_sel, X) * scale
max_abs_diff = (ref - Y).abs().max().item()
torch.testing.assert_close(ref, Y, atol=1e-1, rtol=1e-1)
print("Max abs diff:", max_abs_diff)
print("2D case passed")

## 3D case with
print("3D case")
X = torch.randn(B, n, F_in, device=device, dtype=dtype)
W = torch.randn(L * num_layers, F_out, F_in, device=device, dtype=dtype)
Y = torch.randn(B, n, F_out, device=device, dtype=dtype)
indices = torch.randint(low=0, high=L, size=(B,), device=device, dtype=torch.int32)

# Flatten for sgmv: [B, n, F_in] -> [B*n, F_in]
X_flat = X.reshape(-1, F_in)
Y_flat = Y.reshape(-1, F_out)
indices_expanded = indices.repeat_interleave(n)

prepare_metadata_and_call(Y_flat, X_flat, W, indices_expanded, num_layers, layer_idx, L, scale)

# Correctness check vs. PyTorch
idx = indices.to(torch.long) * num_layers + layer_idx
W_sel = W[idx, :, :]
ref = torch.einsum("bfi,bni->bnf", W_sel, X) * scale
max_abs_diff = (ref - Y).abs().max().item()
torch.testing.assert_close(ref, Y, atol=1e-1, rtol=1e-1)
print("Max abs diff:", max_abs_diff)
print("3D case with passed")

# --------------------------------
# No LoRA adapter cases: selected batches should be unchanged
print("No LoRA adapter cases")

# 2D case: rows with indices==L should be unchanged
print("No LoRA 2D")
X = torch.randn(B, F_in, device=device, dtype=dtype)
W = torch.randn(L * num_layers, F_out, F_in, device=device, dtype=dtype)
Y = torch.randn(B, F_out, device=device, dtype=dtype)
indices = torch.randint(low=0, high=L, size=(B,), device=device, dtype=torch.int32)
mask_nolora = torch.zeros(B, dtype=torch.bool, device=device)
mask_nolora[: max(1, B // 4)] = True  # mark a chunk as no-lora
indices[mask_nolora] = L

Y_orig = Y.clone()
prepare_metadata_and_call(Y, X, W, indices, num_layers, layer_idx, L, scale)

idx = indices.to(torch.long) * num_layers + layer_idx
idx_safe = idx.clone()
idx_safe[mask_nolora] = 0  # avoid OOB gather for ref
W_sel = W[idx_safe, :, :]
ref = torch.einsum("bfi,bi->bf", W_sel, X) * scale

if mask_nolora.any():
    torch.testing.assert_close(Y[mask_nolora], torch.zeros_like(Y[mask_nolora]), atol=1e-1, rtol=1e-1)
if (~mask_nolora).any():
    torch.testing.assert_close(Y[~mask_nolora], ref[~mask_nolora], atol=1e-1, rtol=1e-1)
print("No LoRA 2D passed")

# 3D case
print("No LoRA 3D")
X = torch.randn(B, n, F_in, device=device, dtype=dtype)
W = torch.randn(L * num_layers, F_out, F_in, device=device, dtype=dtype)
Y = torch.randn(B, n, F_out, device=device, dtype=dtype)
indices = torch.randint(low=0, high=L, size=(B,), device=device, dtype=torch.int32)
mask_nolora = torch.zeros(B, dtype=torch.bool, device=device)
mask_nolora[:: max(1, B // 5)] = True
indices[mask_nolora] = L

Y_orig = Y.clone()
X_flat = X.reshape(-1, F_in)
Y_flat = Y.reshape(-1, F_out)
indices_expanded = indices.repeat_interleave(n)

prepare_metadata_and_call(Y_flat, X_flat, W, indices_expanded, num_layers, layer_idx, L, scale)

idx = indices.to(torch.long) * num_layers + layer_idx
idx_safe = idx.clone()
idx_safe[mask_nolora] = 0
W_sel = W[idx_safe, :, :]
ref = torch.einsum("bfi,bni->bnf", W_sel, X) * scale

if mask_nolora.any():
    torch.testing.assert_close(Y[mask_nolora], torch.zeros_like(Y[mask_nolora]), atol=1e-1, rtol=1e-1)
if (~mask_nolora).any():
    torch.testing.assert_close(Y[~mask_nolora], ref[~mask_nolora], atol=1e-1, rtol=1e-1)
print("No LoRA 3D passed")