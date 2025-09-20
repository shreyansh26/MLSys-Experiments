import torch
from bgmv_triton import bgmv_triton

B = 16
num_layers = 8
L = 10
layer_idx = 2
F_in = 16
F_out = 1024
scale = 0.25

dtype = torch.float16  # or torch.bfloat16, torch.float32
device = 'cuda'

X = torch.randn(B, F_in, device=device, dtype=dtype)
W = torch.randn(L * num_layers, F_out, F_in, device=device, dtype=dtype)
Y = torch.zeros(B, F_out, device=device, dtype=dtype)
indices = torch.randint(low=0, high=L, size=(B,), device=device, dtype=torch.int32)

bgmv_triton(Y, X, W, indices, num_layers=num_layers, layer_idx=layer_idx, scale=scale)

# Correctness check vs. PyTorch
idx = indices.to(torch.long) * num_layers + layer_idx
# Gather W[idx] : shape [B, F_out, F_in]
W_sel = W[idx, :, :]
# Batched matmul: [B, F_out, F_in] x [B, F_in, 1] -> [B, F_out, 1]
ref = torch.bmm(W_sel, X.unsqueeze(-1)).squeeze(-1) * scale  # [B, F_out]
max_abs_diff = (ref - Y).abs().max().item()
print("Max abs diff:", max_abs_diff)