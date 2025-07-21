import torch
from flash_attn import flash_attn_func
import math

# shape: (B, seq_len, nheads, headdim)
B, seq_len, nheads, headdim = 2, 2048, 16, 64
q = torch.randn(B, seq_len, nheads, headdim, dtype=torch.float16, device="cuda")
k = torch.randn(B, seq_len, nheads, headdim, dtype=torch.float16, device="cuda")
v = torch.randn(B, seq_len, nheads, headdim, dtype=torch.float16, device="cuda")

# causal=True for auto-regressive models
out = flash_attn_func(q, k, v, causal=True)
print(out.shape)  # (2, 2048, 16, 64)