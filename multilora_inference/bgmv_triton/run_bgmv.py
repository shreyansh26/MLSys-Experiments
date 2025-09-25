import torch
from bgmv_triton import bgmv_triton

B = 16
num_layers = 8
L = 10
layer_idx = 2
F_in = 16
F_out = 1024
scale = 0.25
n = 32

dtype = torch.float16  # or torch.bfloat16, torch.float32
device = 'cuda'

## 2D case
print("2D case")
X = torch.randn(B, F_in, device=device, dtype=dtype)
W = torch.randn(L * num_layers, F_out, F_in, device=device, dtype=dtype)
Y = torch.randn(B, F_out, device=device, dtype=dtype)
indices = torch.randint(low=0, high=L, size=(B,), device=device, dtype=torch.int32)

bgmv_triton(Y, X, W, indices, seq_len=1, num_layers=num_layers, layer_idx=layer_idx, scale=scale)

# Correctness check vs. PyTorch
idx = indices.to(torch.long) * num_layers + layer_idx
# Gather W[idx] : shape [B, F_out, F_in]
W_sel = W[idx, :, :]
# Batched matmul: [B, F_out, F_in] x [B, F_in, 1] -> [B, F_out, 1]
ref = torch.einsum("bfi,bi->bf", W_sel, X) * scale  # [B, F_out]
max_abs_diff = (ref - Y).abs().max().item()
torch.testing.assert_close(ref, Y)
print("Max abs diff:", max_abs_diff)

## 3D case with accumulate=False
print("3D case with accumulate=False")
X = torch.randn(B, n, F_in, device=device, dtype=dtype)
W = torch.randn(L * num_layers, F_out, F_in, device=device, dtype=dtype)
Y = torch.randn(B, n, F_out, device=device, dtype=dtype)
indices = torch.randint(low=0, high=L, size=(B,), device=device, dtype=torch.int32)

Y_orig = Y.clone()
bgmv_triton(Y, X, W, indices, seq_len=n, num_layers=num_layers, layer_idx=layer_idx, scale=scale, accumulate=False)

# Correctness check vs. PyTorch
idx = indices.to(torch.long) * num_layers + layer_idx
# Gather W[idx] : shape [B, F_out, F_in]
W_sel = W[idx, :, :]
# Batched matmul: [B, F_out, F_in] x [B, F_in, 1] -> [B, F_out, 1]
ref = torch.einsum("bfi,bni->bnf", W_sel, X) * scale  # [B, F_out]
max_abs_diff = (ref - Y).abs().max().item()
torch.testing.assert_close(ref, Y)
print("Max abs diff:", max_abs_diff)

## 3D case with accumulate=True
print("3D case with accumulate=True")
X = torch.randn(B, n, F_in, device=device, dtype=dtype)
W = torch.randn(L * num_layers, F_out, F_in, device=device, dtype=dtype)
Y = torch.randn(B, n, F_out, device=device, dtype=dtype)
indices = torch.randint(low=0, high=L, size=(B,), device=device, dtype=torch.int32)

Y_orig = Y.clone()
bgmv_triton(Y, X, W, indices, seq_len=n, num_layers=num_layers, layer_idx=layer_idx, scale=scale, accumulate=True)

# Correctness check vs. PyTorch (accumulate semantics: Y_out = Y_orig + ref)
idx = indices.to(torch.long) * num_layers + layer_idx
# Gather W[idx] : shape [B, F_out, F_in]
W_sel = W[idx, :, :]
# Batched matmul: [B, F_out, F_in] x [B, F_in, 1] -> [B, F_out, 1]
ref = torch.einsum("bfi,bni->bnf", W_sel, X) * scale  # [B, F_out]
max_abs_diff = (Y_orig + ref - Y).abs().max().item()
torch.testing.assert_close(Y_orig + ref, Y, atol=1e-3, rtol=1e-3)
print("Max abs diff:", max_abs_diff)